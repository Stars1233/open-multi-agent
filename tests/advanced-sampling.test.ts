import { describe, it, expect, vi } from 'vitest'
import { OpenMultiAgent } from '../src/index.js'
import type { AgentConfig, LLMChatOptions, LLMMessage, LLMResponse } from '../src/types.js'

let capturedOptions: LLMChatOptions | undefined

vi.mock('../src/llm/adapter.js', () => ({
  createAdapter: async () => {
    return {
      name: 'mock',
      async chat(_msgs: LLMMessage[], options: LLMChatOptions): Promise<LLMResponse> {
        capturedOptions = options
        return {
          id: 'test',
          content: [{ type: 'text', text: 'test response' }],
          model: options.model ?? 'mock',
          stop_reason: 'end_turn',
          usage: { input_tokens: 10, output_tokens: 10 }
        }
      },
      async *stream() {
        yield { type: 'done', data: {} }
      }
    }
  }
}))

describe('Advanced Sampling Parameters', () => {
  it('should map sampling parameters from AgentConfig to LLMChatOptions', async () => {
    const oma = new OpenMultiAgent()

    const config: AgentConfig = {
      name: 'sampler',
      model: 'mock',
      topP: 0.9,
      topK: 40,
      minP: 0.05,
      frequencyPenalty: 1.2,
      presencePenalty: 0.5,
      extraBody: { custom: 'value' },
    }

    await oma.runAgent(config, 'hello')

    expect(capturedOptions).toBeDefined()
    expect(capturedOptions?.topP).toBe(0.9)
    expect(capturedOptions?.topK).toBe(40)
    expect(capturedOptions?.minP).toBe(0.05)
    expect(capturedOptions?.frequencyPenalty).toBe(1.2)
    expect(capturedOptions?.presencePenalty).toBe(0.5)
    expect(capturedOptions?.extraBody).toEqual({ custom: 'value' })
  })
})