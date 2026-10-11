import type { EmbeddingServiceConfig } from "./config.js";
import { deterministicEmbedding, normalizeVector, validateVector } from "./vector.js";

export interface EmbeddingProvider {
  readonly model: string;
  readonly embeddingVersion: string;
  readonly dtype: string;
  readonly dimension: number;
  embed(input: string[]): Promise<number[][]>;
}

export type EmbeddingExtractor = (
  input: string | string[],
  options: Record<string, unknown>,
) => Promise<unknown>;

export function splitModelBatches(
  input: string[],
  maxBatchSize: number,
  maxBatchInputChars: number,
): string[][] {
  const batches: string[][] = [];
  const safeBatchSize = Math.max(1, maxBatchSize);
  const safeMaxBatchInputChars = Math.max(1, maxBatchInputChars);
  let batch: string[] = [];
  let longestInputChars = 0;

  for (const text of input) {
    const inputChars = Array.from(text).length;
    if (
      batch.length > 0 &&
      (batch.length >= safeBatchSize ||
        Math.max(longestInputChars, inputChars) > safeMaxBatchInputChars)
    ) {
      batches.push(batch);
      batch = [];
      longestInputChars = 0;
    }

    batch.push(text);
    longestInputChars = Math.max(longestInputChars, inputChars);
  }

  if (batch.length > 0) {
    batches.push(batch);
  }
  return batches;
}

export class DeterministicEmbeddingProvider implements EmbeddingProvider {
  readonly model: string;
  readonly embeddingVersion: string;
  readonly dtype: string;
  readonly dimension: number;

  constructor(config: EmbeddingServiceConfig) {
    this.model = config.model;
    this.embeddingVersion = config.embeddingVersion;
    this.dtype = config.dtype;
    this.dimension = config.dimension;
  }

  async embed(input: string[]): Promise<number[][]> {
    return input.map((text) => deterministicEmbedding(text, this.dimension));
  }
}

export class TransformersEmbeddingProvider implements EmbeddingProvider {
  readonly model: string;
  readonly embeddingVersion: string;
  readonly dtype: string;
  readonly dimension: number;
  private extractorPromise?: Promise<EmbeddingExtractor>;
  private inferenceTail: Promise<void> = Promise.resolve();

  constructor(
    private readonly config: EmbeddingServiceConfig,
    private readonly extractorLoader: () => Promise<EmbeddingExtractor> = () =>
      loadTransformersExtractor(config),
  ) {
    this.model = config.model;
    this.embeddingVersion = config.embeddingVersion;
    this.dtype = config.dtype;
    this.dimension = config.dimension;
  }

  async embed(input: string[]): Promise<number[][]> {
    if (input.length === 0) {
      return [];
    }
    const extractor = await this.loadExtractor();
    return this.runInference(async () => {
      const vectors: number[][] = [];
      for (const batch of splitModelBatches(
        input,
        this.config.modelBatchSize,
        this.config.modelBatchMaxInputChars,
      )) {
        const output = await extractor(batch, { pooling: "mean", normalize: true });
        try {
          vectors.push(
            ...tensorToVectors(output, batch.length, this.dimension).map((vector) => {
              validateVector(vector, this.dimension);
              return normalizeVector(vector);
            }),
          );
        } finally {
          await disposeEmbeddingOutput(output);
        }
      }
      return vectors;
    });
  }

  private loadExtractor(): Promise<EmbeddingExtractor> {
    if (!this.extractorPromise) {
      this.extractorPromise = this.extractorLoader();
    }
    return this.extractorPromise;
  }

  private async runInference<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.inferenceTail;
    let release!: () => void;
    this.inferenceTail = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }
}

async function disposeEmbeddingOutput(output: unknown): Promise<void> {
  if (
    output &&
    typeof output === "object" &&
    "dispose" in output &&
    typeof output.dispose === "function"
  ) {
    await output.dispose();
  }
}

async function loadTransformersExtractor(
  config: EmbeddingServiceConfig,
): Promise<EmbeddingExtractor> {
  const moduleName = "@huggingface/transformers";
  const transformers = (await import(moduleName)) as {
    env?: {
      cacheDir?: string;
      localModelPath?: string;
      allowLocalModels?: boolean;
    };
    pipeline: (
      task: string,
      model: string,
      options?: Record<string, unknown>,
    ) => Promise<EmbeddingExtractor>;
  };

  if (config.cacheDir && transformers.env) {
    transformers.env.cacheDir = config.cacheDir;
  }
  if (config.localModelPath && transformers.env) {
    transformers.env.localModelPath = config.localModelPath;
    transformers.env.allowLocalModels = true;
  }

  return transformers.pipeline("feature-extraction", config.modelId, {
    dtype: config.dtype,
  });
}

export function tensorToVectors(
  output: unknown,
  expectedCount: number,
  dimension: number,
): number[][] {
  if (expectedCount <= 0) {
    return [];
  }
  const rows = rowsFromTensorOutput(output, expectedCount, dimension);
  if (rows.length !== expectedCount || rows.some((row) => row.length !== dimension)) {
    throw new Error("embedding_output_shape_mismatch");
  }
  return rows;
}

function rowsFromTensorOutput(
  output: unknown,
  expectedCount: number,
  dimension: number,
): number[][] {
  if (Array.isArray(output)) {
    if (expectedCount === 1 && output.every((value) => typeof value === "number")) {
      return [output.map(Number)];
    }
    if (output.every((value) => Array.isArray(value))) {
      return output.map((value) => flattenNumericArray(value));
    }
    throw new Error("unsupported_transformers_embedding_output");
  }
  if (output && typeof output === "object") {
    const candidate = output as {
      data?: Iterable<number>;
      dims?: number[];
      tolist?: () => unknown;
    };
    if (typeof candidate.tolist === "function") {
      return rowsFromTensorOutput(candidate.tolist(), expectedCount, dimension);
    }
    if (candidate.data) {
      const data = Array.from(candidate.data, Number);
      const dims = candidate.dims ?? [];
      if (
        data.length !== expectedCount * dimension ||
        (dims.length > 0 &&
          (dims.length !== 2 || dims[0] !== expectedCount || dims[1] !== dimension))
      ) {
        throw new Error("embedding_output_shape_mismatch");
      }
      return Array.from({ length: expectedCount }, (_, index) =>
        data.slice(index * dimension, (index + 1) * dimension),
      );
    }
  }
  throw new Error("unsupported_transformers_embedding_output");
}

function flattenNumericArray(value: unknown): number[] {
  if (!Array.isArray(value)) {
    if (typeof value === "number") {
      return [value];
    }
    throw new Error("unsupported_transformers_embedding_output");
  }
  const flattened: number[] = [];
  for (const item of value) {
    if (Array.isArray(item)) {
      flattened.push(...flattenNumericArray(item));
    } else if (typeof item === "number") {
      flattened.push(item);
    } else {
      throw new Error("unsupported_transformers_embedding_output");
    }
  }
  return flattened;
}

export function createEmbeddingProvider(config: EmbeddingServiceConfig): EmbeddingProvider {
  if (config.provider === "deterministic") {
    return new DeterministicEmbeddingProvider(config);
  }
  return new TransformersEmbeddingProvider(config);
}
