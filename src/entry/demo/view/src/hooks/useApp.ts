import { useState, useEffect, useCallback } from 'react';
import type {
    Config,
    SystemStatus,
    AnnotatedSample,
    PseudoLabel,
    ModelCheckpoint,
    LoadingState,
    AnnotationData
} from '../models';
import { apiService } from '../services/api';
import { createImageFromBackground, downloadFile } from '../commons/utils.ts';

export const useApp = () => {
    // State
    const [config, setConfig] = useState<Config>({
        budget: 10,
        model_ckpt: 'init_model.pth',
        device: 'cpu',
        batch_size: 4,
        loaded_feature_weight: 1,
        sharp_factor: 1,
        loaded_feature_only: false
    });

    const [status, setStatus] = useState<SystemStatus>({
        train_set_size: 0,
        pool_set_size: 0,
        selected_set_size: 0,
        annotated_set_size: 0,
        current_dataset: 'dataset',
        feature_dict_loaded: false
    });

    const [trainFiles, setTrainFiles] = useState<FileList | null>(null);
    const [poolFiles, setPoolFiles] = useState<FileList | null>(null);
    const [selectedSamples, setSelectedSamples] = useState<string[]>([]);
    const [annotatedSamples, setAnnotatedSamples] = useState<AnnotatedSample[]>([]);
    const [selectedImageIndex, setSelectedImageIndex] = useState<number | null>(null);
    const [pseudoLabel, setPseudoLabel] = useState<PseudoLabel | null>(null);
    const [isAnnotating, setIsAnnotating] = useState(false);
    const [loading, setLoading] = useState<LoadingState>({});
    const [error, setError] = useState<string>('');
    const [success, setSuccess] = useState<string>('');
    const [availableCheckpoints, setAvailableCheckpoints] = useState<ModelCheckpoint[]>([]);
    const [loadingCheckpoints, setLoadingCheckpoints] = useState(false);
    const [brushColor, setBrushColor] = useState('#ff0000');
    const [maskData, setMaskData] = useState<number[][]>([]);

    // Utility functions
    const showError = useCallback((message: string) => {
        setError(message);
        setTimeout(() => setError(''), 5000);
    }, []);

    const showSuccess = useCallback((message: string) => {
        setSuccess(message);
        setTimeout(() => setSuccess(''), 3000);
    }, []);

    // API calls
    const loadStatus = useCallback(async () => {
        try {
            const data = await apiService.getStatus();
            setStatus(data);
        } catch (err) {
            showError('Failed to load status');
        }
    }, [showError]);

    const loadConfig = useCallback(async () => {
        try {
            const data = await apiService.getConfig();
            setConfig(data);
        } catch (err) {
            showError('Failed to load configuration');
        }
    }, [showError]);

    const loadAvailableCheckpoints = useCallback(async () => {
        try {
            setLoadingCheckpoints(true);
            const data = await apiService.getModelCheckpoints();
            console.log(data);
            setAvailableCheckpoints(data.models);
        } catch (err) {
            showError('Failed to load available model checkpoints');
            console.error('Error loading checkpoints:', err);
        } finally {
            setLoadingCheckpoints(false);
        }
    }, [showError]);

    const updateConfig = useCallback((newConfig: Partial<Config>) => {
        setConfig(prev => ({ ...prev, ...newConfig }));
    }, []);

    const uploadFiles = useCallback(async (files: FileList, type: 'train' | 'pool') => {
        try {
            setLoading(prev => ({ ...prev, [type]: true }));

            const result = type === 'train'
                ? await apiService.uploadTrainImages(files)
                : await apiService.uploadPoolImages(files);

            showSuccess(result.message);
            await loadStatus();
        } catch (err) {
            showError(`Failed to upload ${type} files`);
        } finally {
            setLoading(prev => ({ ...prev, [type]: false }));
        }
    }, [showError, showSuccess, loadStatus]);

    const selectSamples = useCallback(async () => {
        try {
            setLoading(prev => ({ ...prev, select: true }));
            await apiService.updateConfig(config);
            const result = await apiService.selectSamples();
            setSelectedSamples(result.selected_images);
            showSuccess('Sample selection completed');
            await loadStatus();
        } catch (err) {
            showError('Failed to select samples');
        } finally {
            setLoading(prev => ({ ...prev, select: false }));
        }
    }, [updateConfig, config, showError, showSuccess, loadStatus]);

    const loadPseudoLabel = useCallback(async (imagePath: string) => {
        try {
            setLoading(prev => ({ ...prev, pseudo: true }));
            const result = await apiService.getPseudoLabel(imagePath);
            setPseudoLabel(result);

            // Initialize mask data with zeros
            const height = result.background.length;
            const width = result.background[0].length;
            setMaskData(Array(height).fill(null).map(() => Array(width).fill(0)));
        } catch (err) {
            showError('Failed to load pseudo label');
        } finally {
            setLoading(prev => ({ ...prev, pseudo: false }));
        }
    }, [showError]);

    const submitAnnotation = useCallback(async () => {
        if (!pseudoLabel || selectedImageIndex === null) return;

        try {
            setLoading(prev => ({ ...prev, annotate: true }));

            const backgroundBase64 = createImageFromBackground(pseudoLabel.background);

            const annotation: AnnotationData = {
                image_path: selectedSamples[selectedImageIndex],
                mask_data: maskData,
                background_image: backgroundBase64
            };

            const result = await apiService.submitAnnotation(annotation);
            showSuccess(result.message);
            await loadAnnotatedSamples();
            await loadStatus();

            // Move to next image or close annotation
            if (selectedImageIndex < selectedSamples.length - 1) {
                setSelectedImageIndex(selectedImageIndex + 1);
                await loadPseudoLabel(selectedSamples[selectedImageIndex + 1]);
            } else {
                setIsAnnotating(false);
                setSelectedImageIndex(null);
                setPseudoLabel(null);
            }
        } catch (err) {
            showError('Failed to submit annotation');
        } finally {
            setLoading(prev => ({ ...prev, annotate: false }));
        }
    }, [pseudoLabel, selectedImageIndex, selectedSamples, maskData, showError, showSuccess, loadPseudoLabel]);

    const loadAnnotatedSamples = useCallback(async () => {
        try {
            const result = await apiService.getAnnotatedSamples();
            setAnnotatedSamples(result.annotated_samples);
        } catch (err) {
            showError('Failed to load annotated samples');
        }
    }, [showError]);

    const downloadDataset = useCallback(async () => {
        try {
            setLoading(prev => ({ ...prev, download: true }));
            const blob = await apiService.downloadDataset();
            downloadFile(blob, 'dataset.zip');
            showSuccess('Dataset downloaded successfully');
        } catch (err) {
            showError('Failed to download dataset');
        } finally {
            setLoading(prev => ({ ...prev, download: false }));
        }
    }, [showError, showSuccess]);

    const resetSystem = useCallback(async () => {
        try {
            setLoading(prev => ({ ...prev, reset: true }));
            await apiService.resetSystem();

            // Reset local state
            setSelectedSamples([]);
            setAnnotatedSamples([]);
            setSelectedImageIndex(null);
            setPseudoLabel(null);
            setIsAnnotating(false);
            setTrainFiles(null);
            setPoolFiles(null);

            showSuccess('System reset successfully');
            loadStatus();
        } catch (err) {
            showError('Failed to reset system');
        } finally {
            setLoading((prev: any) => ({ ...prev, reset: false }));
        }
    }, [showError, showSuccess, loadStatus]);

    const startAnnotation = useCallback((index: number) => {
        setSelectedImageIndex(index);
        setIsAnnotating(true);
        loadPseudoLabel(selectedSamples[index]);
    }, [selectedSamples, loadPseudoLabel]);

    // Initialize data on mount
    useEffect(() => {
        loadStatus();
        loadConfig();
        loadAvailableCheckpoints();
        loadAnnotatedSamples();
    }, [loadStatus, loadConfig, loadAvailableCheckpoints, loadAnnotatedSamples]);

    return {
        // State
        config,
        status,
        trainFiles,
        poolFiles,
        selectedSamples,
        annotatedSamples,
        selectedImageIndex,
        pseudoLabel,
        isAnnotating,
        loading,
        error,
        success,
        availableCheckpoints,
        loadingCheckpoints,
        brushColor,
        maskData,

        // Actions
        setTrainFiles,
        setPoolFiles,
        setBrushColor,
        updateConfig,
        uploadFiles,
        selectSamples,
        submitAnnotation,
        downloadDataset,
        resetSystem,
        startAnnotation,
        loadAvailableCheckpoints,
        showError,
        showSuccess
    };
};
