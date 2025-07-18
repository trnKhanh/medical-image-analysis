export interface Config {
    budget: number;
    model_ckpt: string;
    device: string;
    batch_size: number;
    loaded_feature_weight: number;
    sharp_factor: number;
    loaded_feature_only: boolean;
}

export interface ActiveLearningState {
    train_count: number;
    pool_count: number;
    annotated_count: number;
    selected_count: number;
}

export interface AnnotatedSample {
    path: string;
    visual: string;
}

export interface PseudoLabel {
    background: number[][][];
    layers: number[][][][];
    image_content: string;
}

export interface AnnotationData {
    image_path: string;
    mask_data: number[][];
    background_image: string;
}

export interface ModelCheckpoint {
    name: string;
    size: number;
    description: string;
    create_at: string;
}

export interface CheckpointResponse {
    models: ModelCheckpoint[];
    total_count: number;
}

export interface LoadingState {
    [key: string]: boolean;
}

export interface ApiResponse<T = any> {
    message?: string;
    data?: T;
}

export interface SelectionResponse {
    selected_images: SelectedSample[];
}

export interface SelectedSample {
    path: string;
    name: string;
    data: string;
}
