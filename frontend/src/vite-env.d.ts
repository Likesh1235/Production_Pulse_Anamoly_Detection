/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
  readonly VITE_MAX_BATCH_SIZE?: string;
  readonly VITE_REQUEST_TIMEOUT_MS?: string;
  readonly VITE_STATUS_POLL_MS?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

/** Injected by vite.config.ts from package.json. */
declare const __APP_VERSION__: string;
