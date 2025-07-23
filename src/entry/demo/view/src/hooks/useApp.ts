import { useState, useEffect, useCallback } from 'react';
import type {
    Config,
    AnnotatedSample,
    PseudoLabel,
    ModelCheckpoint,
    LoadingState,
    ActiveLearningState, SelectedSample
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

    const [status, setStatus] = useState<ActiveLearningState>({
        train_count: 0,
        annotated_count: 0,
        pool_count: 0,
        selected_count: 0
    });

    const [trainFiles, setTrainFiles] = useState<FileList | null>(null);
    const [poolFiles, setPoolFiles] = useState<FileList | null>(null);
    const [selectedSamples, setSelectedSamples] = useState<SelectedSample[]>([]);
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
            console.error(err);
        }
    }, [showError]);

    const loadConfig = useCallback(async () => {
        try {
            const data = await apiService.getConfig();
            setConfig(data);
        } catch (err) {
            showError('Failed to load configuration');
            console.error(err);
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

            const result = await apiService.uploadImages(files, type)

            showSuccess(result.message);
            await loadStatus();
        } catch (err) {
            showError(`Failed to upload ${type} files`);
            console.error(err);
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
            console.log(result.selected_images);
            showSuccess('Sample selection completed');
            await loadStatus();
        } catch (err) {
            showError('Failed to select samples');
            console.error(err);
        } finally {
            setLoading(prev => ({ ...prev, select: false }));
        }
    }, [updateConfig, config, showError, showSuccess, loadStatus]);

    const loadPseudoLabel = useCallback(async (imageIndex: number) => {
        try {
            setLoading(prev => ({ ...prev, pseudo: true }));
            const result = await apiService.getPseudoLabel(imageIndex);
            console.log("PSEUDO LABEL:", result);
            setPseudoLabel(result);
            const height = result.background.length;
            const width = result.background[0].length;
            setMaskData(Array(height).fill(null).map(() => Array(width).fill(0)));
        } catch (err) {
            showError('Failed to load pseudo label');
            console.error(err);
        } finally {
            setLoading(prev => ({ ...prev, pseudo: false }));
        }
    }, [showError]);

    const submitAnnotation = useCallback(async (annotationData: any) => {
        if (!pseudoLabel || selectedImageIndex === null) return;

        try {
            setLoading(prev => ({ ...prev, annotate: true }));

            const backgroundBase64 = createImageFromBackground(pseudoLabel.background);

            const annotation = {
                image_index: selectedImageIndex,
                layers: [annotationData.annotation_mask],
                background: backgroundBase64
            };

            const response = await apiService.submitAnnotation(annotation);

            showSuccess(response.message);
            await loadAnnotatedSamples();
            await loadStatus();

            if (selectedImageIndex < selectedSamples.length - 1) {
                setSelectedImageIndex(selectedImageIndex + 1);
                await loadPseudoLabel(selectedImageIndex + 1);
            } else {
                setIsAnnotating(false);
                setSelectedImageIndex(null);
                setPseudoLabel(null);
            }
        } catch (err) {
            showError('Failed to submit annotation');
            console.error(err);
        } finally {
            setLoading(prev => ({ ...prev, annotate: false }));
        }
    }, [
        pseudoLabel,
        selectedImageIndex,
        selectedSamples,
        showError,
        showSuccess,
        loadPseudoLabel
    ]);

    const loadAnnotatedSamples = useCallback(async () => {
        try {
            const result = await apiService.getAnnotatedSamples();
            setAnnotatedSamples(result.annotated_samples);
        } catch (err) {
            showError(`Failed to load annotated samples`);
            console.error(err);
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
            console.error(err);
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
            console.error(err)
        } finally {
            setLoading((prev: any) => ({ ...prev, reset: false }));
        }
    }, [showError, showSuccess, loadStatus]);

    const syncSystem = useCallback(async () => {
        console.log('syncSystem');
    }, []);

    const startAnnotation = useCallback((index: number) => {
        setSelectedImageIndex(index);
        setIsAnnotating(true);
        loadPseudoLabel(index);
    }, [selectedSamples, loadPseudoLabel]);

    const cancelAnnotation = () => {
        setIsAnnotating(false);
        setSelectedImageIndex(null);
        setPseudoLabel(null);
    };

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
        syncSystem,
        startAnnotation,
        cancelAnnotation,
        loadAvailableCheckpoints,
        showError,
        showSuccess
    };
};
