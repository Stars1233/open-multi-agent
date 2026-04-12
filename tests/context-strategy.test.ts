import { describe, it, expect, vi } from 'vitest'
import { z } from 'zod'
import { AgentRunner } from '../src/agent/runner.js'
import { ToolRegistry, defineTool } from '../src/tool/framework.js'
import { ToolExecutor } from '../src/tool/executor.js'
import type { LLMAdapter, LLMChatOptions, LLMMessage, LLMResponse, TraceEvent } from '../src/types.js'

function textResponse(text: string): LLMResponse {
  return {
    id: `resp-${Math.random().toString(36).slice(2)}`,
    content: [{ type: 'text', text }],
    model: 'mock-model',
    stop_reason: 'end_turn',
    usage: { input_tokens: 10, output_tokens: 20 },
  }
}

function toolUseResponse(toolName: string, input: Record<string, unknown>): LLMResponse {
  return {
    id: `resp-${Math.random().toString(36).slice(2)}`,
    content: [{
      type: 'tool_use',
      id: `tu-${Math.random().toString(36).slice(2)}`,
      name: toolName,
      input,
    }],
    model: 'mock-model',
    stop_reason: 'tool_use',
    usage: { input_tokens: 15, output_tokens: 25 },
  }
}

function buildRegistryAndExecutor(): { registry: ToolRegistry; executor: ToolExecutor } {
  const registry = new ToolRegistry()
  registry.register(
    defineTool({
      name: 'echo',
      description: 'Echo input',
      inputSchema: z.object({ message: z.string() }),
      async execute({ message }) {
        return { data: message }
      },
    }),
  )
  return { registry, executor: new ToolExecutor(registry) }
}

describe('AgentRunner contextStrategy', () => {
  it('keeps baseline behavior when contextStrategy is not set', async () => {
    const calls: LLMMessage[][] = []
    const adapter: LLMAdapter = {
      name: 'mock',
      async chat(messages) {
        calls.push(messages.map(m => ({ role: m.role, content: m.content })))
        return calls.length === 1
          ? toolUseResponse('echo', { message: 'hello' })
          : textResponse('done')
      },
      async *stream() {
        /* unused */
      },
    }
    const { registry, executor } = buildRegistryAndExecutor()
    const runner = new AgentRunner(adapter, registry, executor, {
      model: 'mock-model',
      allowedTools: ['echo'],
      maxTurns: 4,
    })

    await runner.run([{ role: 'user', content: [{ type: 'text', text: 'start' }] }])
    expect(calls).toHaveLength(2)
    expect(calls[0]).toHaveLength(1)
    expect(calls[1]!.length).toBeGreaterThan(calls[0]!.length)
  })

  it('sliding-window truncates old turns and preserves the first user message', async () => {
    const calls: LLMMessage[][] = []
    const responses = [
      toolUseResponse('echo', { message: 't1' }),
      toolUseResponse('echo', { message: 't2' }),
      toolUseResponse('echo', { message: 't3' }),
      textResponse('done'),
    ]
    let idx = 0
    const adapter: LLMAdapter = {
      name: 'mock',
      async chat(messages) {
        calls.push(messages.map(m => ({ role: m.role, content: m.content })))
        return responses[idx++]!
      },
      async *stream() {
        /* unused */
      },
    }
    const { registry, executor } = buildRegistryAndExecutor()
    const runner = new AgentRunner(adapter, registry, executor, {
      model: 'mock-model',
      allowedTools: ['echo'],
      maxTurns: 8,
      contextStrategy: { type: 'sliding-window', maxTurns: 1 },
    })

    await runner.run([{ role: 'user', content: [{ type: 'text', text: 'original prompt' }] }])

    const laterCall = calls[calls.length - 1]!
    const firstUserText = laterCall[0]!.content[0]
    expect(firstUserText).toMatchObject({ type: 'text', text: 'original prompt' })
    const flattenedText = laterCall.flatMap(m => m.content.filter(c => c.type === 'text'))
    expect(flattenedText.some(c => c.type === 'text' && c.text.includes('truncated'))).toBe(true)
  })

  it('summarize strategy replaces old context and emits summary trace call', async () => {
    const calls: Array<{ messages: LLMMessage[]; options: LLMChatOptions }> = []
    const traces: TraceEvent[] = []
    const responses = [
      toolUseResponse('echo', { message: 'first turn payload '.repeat(20) }),
      toolUseResponse('echo', { message: 'second turn payload '.repeat(20) }),
      textResponse('This is a concise summary.'),
      textResponse('final answer'),
    ]
    let idx = 0
    const adapter: LLMAdapter = {
      name: 'mock',
      async chat(messages, options) {
        calls.push({ messages: messages.map(m => ({ role: m.role, content: m.content })), options })
        return responses[idx++]!
      },
      async *stream() {
        /* unused */
      },
    }
    const { registry, executor } = buildRegistryAndExecutor()
    const runner = new AgentRunner(adapter, registry, executor, {
      model: 'mock-model',
      allowedTools: ['echo'],
      maxTurns: 8,
      contextStrategy: { type: 'summarize', maxTokens: 20 },
    })

    const result = await runner.run(
      [{ role: 'user', content: [{ type: 'text', text: 'start' }] }],
      { onTrace: (e) => { traces.push(e) }, runId: 'run-summary', traceAgent: 'context-agent' },
    )

    const summaryCall = calls.find(c => c.messages.length === 1 && c.options.tools === undefined)
    expect(summaryCall).toBeDefined()
    const llmTraces = traces.filter(t => t.type === 'llm_call')
    expect(llmTraces.some(t => t.type === 'llm_call' && t.phase === 'summary')).toBe(true)

    // Summary adapter usage must count toward RunResult.tokenUsage (maxTokenBudget).
    expect(result.tokenUsage.input_tokens).toBe(15 + 15 + 10 + 10)
    expect(result.tokenUsage.output_tokens).toBe(25 + 25 + 20 + 20)

    // After compaction, summary text is folded into the next user turn (not a
    // standalone user message), preserving user/assistant alternation.
    const turnAfterSummary = calls.find(
      c => c.messages.some(
        m => m.role === 'user' && m.content.some(
          b => b.type === 'text' && b.text.includes('[Conversation summary]'),
        ),
      ),
    )
    expect(turnAfterSummary).toBeDefined()
    const rolesAfterFirstUser = turnAfterSummary!.messages.map(m => m.role).join(',')
    expect(rolesAfterFirstUser).not.toMatch(/^user,user/)
  })

  it('custom strategy calls compress callback and uses returned messages', async () => {
    const compress = vi.fn((messages: LLMMessage[]) => messages.slice(-1))
    const calls: LLMMessage[][] = []
    const responses = [
      toolUseResponse('echo', { message: 'hello' }),
      textResponse('done'),
    ]
    let idx = 0
    const adapter: LLMAdapter = {
      name: 'mock',
      async chat(messages) {
        calls.push(messages.map(m => ({ role: m.role, content: m.content })))
        return responses[idx++]!
      },
      async *stream() {
        /* unused */
      },
    }
    const { registry, executor } = buildRegistryAndExecutor()
    const runner = new AgentRunner(adapter, registry, executor, {
      model: 'mock-model',
      allowedTools: ['echo'],
      maxTurns: 4,
      contextStrategy: {
        type: 'custom',
        compress,
      },
    })

    await runner.run([{ role: 'user', content: [{ type: 'text', text: 'custom prompt' }] }])

    expect(compress).toHaveBeenCalledOnce()
    expect(calls[1]).toHaveLength(1)
  })
})
