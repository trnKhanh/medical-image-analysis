import axios, {type AxiosInstance, type AxiosRequestConfig, type AxiosResponse } from 'axios';
import type {
    Config,
    CheckpointResponse,
    SelectionResponse,
    AnnotatedSample,
    PseudoLabel,
    DatasetState, AnnotationData, SelectedSample, DiskInfo,
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

apiClient.interceptors.response.use(
    (res) => res,
    (err) => {
        console.error('API call failed:', err);
        console.error('Response data:', err.response?.data);
        console.error('Request config:', err.config);

        let errorMessage = err.message;
        if (err.response?.data) {
            if (typeof err.response.data === 'object') {
                errorMessage = `HTTP ${err.response.status}: ${JSON.stringify(err.response.data, null, 2)}`;
            } else {
                errorMessage = `HTTP ${err.response.status}: ${err.response.data}`;
            }
        }

        return Promise.reject(new Error(errorMessage));
    },
);

apiClient.interceptors.request.use(
    (config) => {
        const workspace = ApiService.getCurrentWorkspace();
        if (workspace) {
            config.headers['X-Workspace'] = workspace;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

class ApiService {
    private static currentWorkspace: string | null = null;

    /**
     * Extract workspace from current URL path
     * e.g., http://localhost:3333/abc -> "abc"
     */
    static getCurrentWorkspace(): string | null {
        if (typeof window !== 'undefined') {
            const path = window.location.pathname;
            const segments = path.split('/').filter(Boolean);
            return segments[0] || null;
        }
        return this.currentWorkspace;
    }

    /**
     * Set workspace manually (useful for SSR or testing)
     */
    static setWorkspace(workspace: string): void {
        this.currentWorkspace = workspace;
    }

    /**
     * Clear the current workspace
     */
    static clearWorkspace(): void {
        this.currentWorkspace = null;
    }

    private async request<T = unknown>(
        cfg: AxiosRequestConfig,
    ): Promise<T> {
        const {
            method = 'GET',
            data,
            headers = {},
            ...rest
        } = cfg;

        const finalHeaders =
            data && !(data instanceof FormData)
                ? { 'Content-Type': 'application/json', ...headers }
                : headers;

        const workspace = ApiService.getCurrentWorkspace();
        if (workspace && !finalHeaders['X-Workspace']) {
            finalHeaders['X-Workspace'] = workspace;
        }

        const response: AxiosResponse<T> = await apiClient.request<T>({
            method,
            data,
            headers: finalHeaders,
            ...rest,
        });

        return response.data;
    }

    // ------------------- High‑level convenience methods -----------------------

    getDatasetState(): Promise<DatasetState> {
        return this.request<DatasetState>({ url: '/dataset/state' });
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

        // Add workspace to form data
        const workspace = ApiService.getCurrentWorkspace();
        if (workspace) {
            fd.append('workspace', workspace);
        }

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
            url: '/active-learning/select-samples',
            method: 'POST',
        });
    }

    getPseudoLabel(imagePath: string): Promise<PseudoLabel> {
        return this.request<PseudoLabel>({
            url: `/active-learning/pseudo-label`,
            method: 'GET',
            params: {
                image_path: imagePath
            }
        });
    }

    submitAnnotation(annotation: AnnotationData): Promise<{ message: string }> {
        const formData = new FormData();
        formData.append("image_path", annotation.image_path);
        formData.append("background", annotation.background);
        formData.append("layers", JSON.stringify(annotation.layers));

        // Add workspace to annotation
        const workspace = ApiService.getCurrentWorkspace();
        if (workspace) {
            formData.append("workspace", workspace);
        }

        return this.request({
            url: '/active-learning/annotate',
            method: 'POST',
            data: formData,
        });
    }

    getAnnotatedSamples(): Promise<{ annotated_samples: AnnotatedSample[] }> {
        return this.request({
            url: '/active-learning/annotated',
        });
    }

    getSelectedSamples(): Promise<{ selected_samples: SelectedSample[] }> {
        return this.request({
            url: '/active-learning/selected-samples',
        });
    }

    async downloadDataset(): Promise<Blob> {
        const workspace = ApiService.getCurrentWorkspace();
        const params = workspace ? { workspace } : {};

        const { data } = await apiClient.get<Blob>('/dataset/download', {
            responseType: 'blob',
            params,
        });
        return data;
    }

    resetSystem(): Promise<void> {
        return this.request<void>({
            url: '/reset',
            method: 'POST',
        });
    }

    setWorkspace(workspace: string): void {
        ApiService.setWorkspace(workspace);
    }

    getDiskInfo(): Promise<DiskInfo> {
        return this.request<DiskInfo>({ url: '/dataset/disk-info' });
    }
}

export const apiService = new ApiService();
