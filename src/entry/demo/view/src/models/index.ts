export interface Config {
    budget: number;
    model_ckpt: string;
    device: string;
    batch_size: number;
    loaded_feature_weight: number;
    sharp_factor: number;
    loaded_feature_only: boolean;
}

export interface SystemStatus {
    train_set_size: number;
    pool_set_size: number;
    selected_set_size: number;
    annotated_set_size: number;
    current_dataset: string;
    feature_dict_loaded: boolean;
}

export interface AnnotatedSample {
    path: string;
    visual: string;
}

export interface PseudoLabel {
    background: number[][][];
    layers: number[][][][];
    image_path: string;
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
    selected_images: string[];
    dataset_id: string;
}
