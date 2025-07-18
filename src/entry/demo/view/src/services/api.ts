import axios, {type AxiosInstance, type AxiosRequestConfig, type AxiosResponse } from 'axios';
import type {
    Config,
    CheckpointResponse,
    SelectionResponse,
    AnnotatedSample,
    PseudoLabel,
    AnnotationData,
    ActiveLearningState,
} from '../models';

const API_BASE_URL = 'http://localhost:8000/api/v1';

/**
 * Create one configured Axios client so every call shares
 * the same baseURL, timeout, auth headers / tokens, interceptors, etc.
 */
const apiClient: AxiosInstance = axios.create({
    baseURL: API_BASE_URL,
    timeout: 100_000,
    headers: { Accept: 'application/json' },
});

/** Convert Axios errors to something easier to read (optional) */
apiClient.interceptors.response.use(
    (res) => res,
    (err) => {
        console.error('API call failed:', err);
        return Promise.reject(
            new Error(
                err.response
                    ? `HTTP ${err.response.status}: ${err.response.data?.detail ?? err.message}`
                    : err.message,
            ),
        );
    },
);

class ApiService {
    private async request<T = unknown>(
        cfg: AxiosRequestConfig,
    ): Promise<T> {
        const {
            method = 'GET',
            data,
            headers = {},
            ...rest
        } = cfg;

        // Only add Content‑Type when we know we’re sending JSON (not FormData)
        const finalHeaders =
            data && !(data instanceof FormData)
                ? { 'Content-Type': 'application/json', ...headers }
                : headers;

        const response: AxiosResponse<T> = await apiClient.request<T>({
            method,
            data,
            headers: finalHeaders,
            ...rest,
        });

        return response.data;
    }

    // ------------------- High‑level convenience methods -----------------------

    getStatus(): Promise<ActiveLearningState> {
        return this.request<ActiveLearningState>({ url: '/active-learning/state' });
    }

    getConfig(): Promise<Config> {
        return this.request<Config>({ url: '/active-learning/config' });
    }

    updateConfig(config: Partial<Config>): Promise<void> {
        return this.request<void>({
            url: '/active-learning/config',
            method: 'POST',
            data: config,
        });
    }

    getModelCheckpoints(): Promise<CheckpointResponse> {
        return this.request<CheckpointResponse>({
            url: '/models/checkpoints',
        });
    }

    private buildFormData(files: FileList, type: string): FormData {
        const fd = new FormData();
        Array.from(files).forEach((f) => fd.append('files', f));
        fd.append('type', type);
        return fd;
    }

    uploadImages(files: FileList, type: string): Promise<{ message: string }> {
        return this.request({
            url: '/dataset/upload/images',
            method: 'POST',
            data: this.buildFormData(files, type),
        });
    }

    selectSamples(): Promise<SelectionResponse> {
        return this.request<SelectionResponse>({
            url: '/active-learning/select/samples',
            method: 'POST',
        });
    }

    getPseudoLabel(imageIndex: number): Promise<PseudoLabel> {
        return this.request<PseudoLabel>({
            url: `/active-learning/pseudo-label/${encodeURIComponent(imageIndex)}`,
        });
    }

    submitAnnotation(annotation: AnnotationData): Promise<{ message: string }> {
        return this.request({
            url: '/active-learning/annotate',
            method: 'POST',
            data: annotation,
        });
    }

    getAnnotatedSamples(): Promise<{ annotated_samples: AnnotatedSample[] }> {
        return this.request({
            url: '/active-learning/annotated',
        });
    }

    async downloadDataset(): Promise<Blob> {
        const { data } = await apiClient.get<Blob>('/dataset/download', {
            responseType: 'blob',
        });
        return data;
    }

    resetSystem(): Promise<void> {
        return this.request<void>({
            url: '/reset',
            method: 'POST',
        });
    }
}

export const apiService = new ApiService();