// API proposal for window.ai low-level extension
// based on https://github.com/explainers-by-googlers/prompt-api

// from existing proposal:
export type AIAssistantPromptRole = "system" | "user" | "assistant";

export interface AIAssistantPrompt {
  role: AIAssistantPromptRole;
  content: string;
}

// new stuff
export interface AITokenizationOptions {
  /**
   * During tokenization, should things like <|endoftext|> be tokenized as
   * the special token, or as separate non-special tokens (eg., "<|" "end" "of" "text" "|>")
   */
  allowSpecial?: boolean;
}

export interface AISequenceOptions {
  /**
   * Initial prompt of the sequence.
   * The final `assistant` message will be added automatically and should not
   * be included here.
   */
  messages: AIAssistantPrompt[];
}

export type AITokenId = number;

export interface AITokenizer {
  /**
   * Vocabulary size of the model - the size of the logits vector.
   */
  vocabSize: number;

  /**
   * When model samples this, the sequence should finish.
   * For chat models, this is end-of-turn token.
   */
  eosToken: AITokenId;

  /**
   * Return the byte representation of the token.
   * Returns an empty array for special tokens like "<|endoftext|>".
   * To de-tokenize an array of tokens, concatenate the bytes of the tokens.
   *
   * @param tokenId the token number, between 0 and vocabSize - 1; throws otherwise
   */
  tokenBytes(tokenId: AITokenId): Uint8Array;

  /**
   * Return the name of the token eg., "the", "▁Comment", "<|endoftext|>", "<0x17>",
   * "лі", "ĠComment"
   *
   * @param tokenId the token number, between 0 and vocabSize - 1; throws otherwise
   */
  tokenName(tokenId: AITokenId): string;

  /**
   * Tokenize (encode) the text without prepending the beginning-of-sequence or spaces,
   * or appending anything. This can used in the middle of a sequence.
   *
   * @param text the text to tokenize
   * @param options options
   */
  tokenizeExact(text: string, options?: AITokenizationOptions): Int32Array;
}

export interface AIModel extends AITokenizer {
  /**
   * Context size of the model.
   */
  maxTokens: number;

  /**
   * Create a new sequence with given prompt.
   * This generally doesn't run the prefill, until the first advance.
   */
  createSequence(options: AISequenceOptions): Promise<AISequence>;
}

export interface AISamplingOptions {
  /**
   * The temperature of the sampling. Higher temperature means more randomness.
   * Default is 0.0 - argmax sampling.
   */
  temperature?: number;

  /**
   * The number of tokens to sample. Defaults to vocab size.
   */
  topK?: number;

  /**
   * If provided, only tokens set in this bitmask can be sampled.
   * The token N is allowed iff `samplingMask[N >> 5] & (1 << (N & 31)) != 0`.
   */
  samplingMask?: Uint32Array;
}

export interface AISequence {
  /**
   * The size of the prompt in tokens.
   */
  promptSize: number;

  /**
   * The output tokens in the sequence
   */
  tokens: readonly AITokenId[];

  /**
   * The maximum number of tokens that can be still generated in the sequence.
   * Typically: model.maxTokens - this.promptSize - this.tokens.length
   */
  tokensLeft: number;

  /**
   * Remove the last `backtrack` tokens from the sequence, append `appendTokens`,
   * and compute logits for the next token.
   * `this.advance([])` is generally a no-op, except at the beginning of the sequence.
   *
   * @param appendTokens the list of tokens to append as output
   * @param backtrack defaults to 0; it may be unsupported by some models
   */
  advance(
    appendTokens: ArrayLike<AITokenId>,
    backtrack?: number,
  ): Promise<void>;

  /**
   * Get the raw logits for the next token.
   * Calls `advance([])` if the logits are not yet computed.
   */
  logits(): Promise<Float32Array>;

  /**
   * Sample from the current logits with given options.
   * Can be called multiple times to sample from the same logits.
   * Calls `advance([])` if the logits are not yet computed.
   *
   * @param options sampling options
   */
  sample(options?: AISamplingOptions): Promise<AITokenId>;

  /**
   * Dispose of the sequence and release resources (KV cache etc.).
   * The sequence cannot be used after this.
   */
  destroy(): void;

  /**
   * Create a copy of the sequence, with the same prompt and output tokens.
   * It can then be advanced independently.
   */
  clone(): Promise<AISequence>;
}

// Example:
export async function genText(
  tokenizer: AITokenizer,
  sequence: AISequence,
  maxTokens = 128,
): Promise<string> {
  const decoder = new TextDecoder();
  let output = "";

  for (let i = 0; i < maxTokens; i++) {
    const token = await sequence.sample();
    if (token === tokenizer.eosToken) {
      break;
    }
    await sequence.advance([token]);
    // decode token bytes to string
    output += decoder.decode(tokenizer.tokenBytes(token), { stream: true });
  }
  // decode remaining bytes if any
  output += decoder.decode();

  // release KV cache etc.
  sequence.destroy();

  return output;
}

// Example with computing mask in parallel:
export async function genTextConstrained(
  tokenizer: AITokenizer,
  sequence: AISequence,
  // user-provided hook to compute temperature and mask, given existing output tokens
  getMask: (
    tokensSoFar: readonly AITokenId[],
  ) => Promise<[number, Uint32Array]>,
  maxTokens = 128,
): Promise<string> {
  const decoder = new TextDecoder();
  let output = "";

  let fwd = sequence.advance([]); // start first forward pass

  for (let i = 0; i < maxTokens; i++) {
    // compute mask in parallel with the forward pass
    const [temperature, samplingMask] = await getMask(sequence.tokens);
    // wait for fwd pass to finish
    await fwd;
    // sample with mask
    const token = await sequence.sample({ samplingMask, temperature });
    if (token === tokenizer.eosToken) {
      break;
    }
    // start next forward pass
    fwd = sequence.advance([token]);
    output += decoder.decode(tokenizer.tokenBytes(token), { stream: true });
  }
  output += decoder.decode();

  sequence.destroy();

  return output;
}
