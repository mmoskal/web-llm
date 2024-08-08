import { MLCEngine } from "./engine";
import * as tvmjs from "tvmjs";
import {
  AISequenceOptions,
  AISequence,
  AIModel,
  AITokenId,
  AITokenizationOptions,
} from "./window_ai_ll_iface";

const TOKENIZER_PREFIX = "\x02";

export function createAIModel(engine: MLCEngine): AIModel {
  return new MlcAIModel(engine);
}

class MlcAIModel implements AIModel {
  private tokPrefix: Int32Array | undefined;
  private tokPrefix0 = 0;

  constructor(public _engine: MLCEngine) {}

  async createSequence(options: AISequenceOptions): Promise<AISequence> {
    return new MlcAiSequence(this._engine, options);
  }

  get maxTokens(): number {
    return this._engine.config!.context_window_size;
  }

  get vocabSize(): number {
    return this._engine.config!.vocab_size;
  }

  get eosToken(): number {
    return this._engine.pipeline!.stopTokens[0];
  }

  private getTokPrefix() {
    if (!this.tokPrefix) {
      this.tokPrefix =
        this._engine.pipeline!.tokenizer.encode(TOKENIZER_PREFIX);
      this.tokPrefix0 = this.tokPrefix[this.tokPrefix.length - 1];
    }
    return this.tokPrefix;
  }

  tokenBytes(tokenId: AITokenId): Uint8Array {
    const tok = this._engine.pipeline!.tokenizer;

    this.getTokPrefix(); // compute tokPrefix0
    const s = tok.decode(Int32Array.from([this.tokPrefix0, tokenId])).slice(1);

    const isSpecial =
      s == "<s>" ||
      s == "</s>" ||
      s == "<pad>" ||
      s == "<unk>" ||
      (s.startsWith("<|") && s.endsWith("|>"));

    if (isSpecial) {
      return new Uint8Array();
    }

    const encoder = new TextEncoder();

    if (s.includes("\uFFFD")) {
      const id = tok.idToToken(tokenId);
      if (id.startsWith("<0x")) {
        return new Uint8Array([parseInt(id.slice(3), 16)]);
      } else if (id == s) {
        return encoder.encode(s);
      } else {
        throw new Error(
          `Invalid token: ${JSON.stringify(s)} ${id} at ${tokenId}`,
        );
      }
    } else {
      return encoder.encode(s);
    }
  }

  tokenName(tokenId: AITokenId): string {
    const tok = this._engine.pipeline!.tokenizer;
    const r = tok.idToToken(tokenId);
    if (r == "" && (tokenId < 0 || tokenId >= this.vocabSize)) {
      throw new Error(`Invalid token id: ${tok}`);
    }
    return r;
  }

  // only encode the exact text
  tokenizeExact(text: string, options?: AITokenizationOptions): Int32Array {
    const tok = this._engine.pipeline!.tokenizer;
    if (this.tokPrefix === undefined) {
      this.tokPrefix = tok.encode(TOKENIZER_PREFIX);
    }

    const r = tok.encode(TOKENIZER_PREFIX + text)!;

    for (let i = 0; i < this.tokPrefix.length; i++) {
      if (r[i] != this.tokPrefix[i]) {
        throw new Error("Tokenizer prefix mismatch");
      }
    }

    return r.slice(this.tokPrefix.length);
  }
}

class MlcAiSequence implements AISequence {
  private _tokens: number[];
  private _promptSize = 0;
  private _logits: tvmjs.NDArray | undefined;

  constructor(
    private _engine: MLCEngine2,
    private _options: AISequenceOptions,
  ) {
    this._tokens = [];
  }

  get promptSize(): number {
    if (this._promptSize == 0)
      throw new Error("Prompt size not initialized yet");
    return this._promptSize;
  }

  get maxTokens(): number {
    return this._engine.config!.context_window_size - this._promptSize;
  }

  get tokensSoFar(): number {
    return this._tokens.length;
  }

  get tokensLeft(): number {
    return this.maxTokens - this.tokensSoFar;
  }

  get tokens(): number[] {
    return this._tokens.slice();
  }

  private _getPipeline() {
    return this._engine.pipeline!;
  }

  async advance(appendTokens: number[], backtrack = 0): Promise<void> {
    if (!this._promptSize) {
      this._engine._window_ai_seq = this;
      this._promptSize = -1;
      this._getPipeline().lowLevel = true;
      const logits = await this._engine.prefill({
        messages: this._options.messages,
      });
      if (!logits) {
        throw new Error("Can't get logits");
      }
      this._promptSize = this._getPipeline().filledKVCacheLength;
      this._logits?.dispose();
      this._logits = logits;
    } else {
      if (this._engine._window_ai_seq !== this) {
        throw new Error("Cannot advance a different AISequence");
      }
    }

    if (backtrack > 0) {
      this._tokens.splice(-backtrack, backtrack);
      throw new Error("Backtracking not implemented");
    }

    if (appendTokens.length > 0) {
      this._tokens.push(...appendTokens);
      this._logits?.dispose();
      this._logits = await this._getPipeline().forwardTokens(appendTokens);
    }
  }

  async clone(): Promise<AISequence> {
    throw new Error("Not implemented");
  }

  async sample(options?: AISamplingOptions): Promise<number> {
    if (!this._logits) {
      throw new Error("No logits");
    }

    const chat = this._getPipeline();
    await this._logits.device.sync();
    chat.tvm.beginScope();
    const clone = chat.tvm.empty(
      this._logits.shape,
      this._logits.dtype,
      this._logits.device,
    );
    clone.copyFrom(this._logits);
    const token = await chat.sampleTokenFromLogits(
      clone,
      {
        temperature: options?.temperature,
      },
      options?.samplingMask,
    );
    chat.tvm.endScope();
    return token;
  }

  async logProbs(mask?: Uint32Array): Promise<Float32Array> {
    const l = await this.logits();
    let max = -Infinity;
    for (let i = 0; i < l.length; i++) {
      if (mask && !isAllowed(mask, i)) {
        l[i] = -Infinity;
      }
      if (l[i] > max) {
        max = l[i];
      }
    }
    let sum = 0;
    for (let i = 0; i < l.length; i++) {
      sum += Math.exp(l[i] - max);
    }
    const logSum = max + Math.log(sum);
    for (let i = 0; i < l.length; i++) {
      l[i] -= logSum;
    }
    return l;
  }

  // 1 not surprised, 100 or more - very surprised
  async surprise(mask: Uint32Array): Promise<number> {
    const l = await this.logits();
    let max = -Infinity;
    for (let i = 0; i < l.length; i++) {
      if (l[i] > max) {
        max = l[i];
      }
    }
    let sum = 0;
    let sumMasked = 0;
    for (let i = 0; i < l.length; i++) {
      sum += Math.exp(l[i] - max);
      if (mask && !isAllowed(mask, i)) {
        sumMasked += Math.exp(l[i] - max);
      }
    }
    return sum / sumMasked;
  }

  async logits(): Promise<Float32Array> {
    if (!this._logits) {
      throw new Error("No logits");
    }

    const r = this._logits.toArray();
    if (r instanceof Float32Array) {
      return r;
    } else {
      throw new Error("Not a Float32Array");
    }
  }

  // release KV cache
  destroy(): void {
    this._logits?.dispose();
    this._logits = undefined;
    if (this._engine._window_ai_seq === this) {
      this._engine._window_ai_seq = undefined;
    }
  }
}

export interface AISamplingOptions {
  temperature?: number;
  samplingMask?: Uint32Array;
}

function isAllowed(mask: Uint32Array, i: number): boolean {
  return mask[i >>> 5] & (1 << (i & 31)) ? true : false;
}

type MLCEngine2 = MLCEngine & { _window_ai_seq?: MlcAiSequence };
