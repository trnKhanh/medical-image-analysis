import { useState, useEffect, useCallback } from 'react';
import type {
    Config,
    AnnotatedSample,
    PseudoLabel,
    ModelCheckpoint,
    LoadingState,
    SelectedSample, AnnotationData, DiskInfo, DatasetState
} from '../models';
import { apiService } from '../services/api';
import { downloadFile } from '../commons/utils.ts';

export const useApp = () => {
    const [config, setConfig] = useState<Config>({
        budget: 10,
        model_ckpt: 'init_model.pth',
        device: 'cpu',
        batch_size: 4,
        loaded_feature_weight: 1,
        sharp_factor: 1,
        loaded_feature_only: false
    });

    const [datasetState, setDatasetState] = useState<DatasetState>({
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

    const [diskInfo, setDiskInfo] = useState<DiskInfo | null>(null);
    const [loadingDiskInfo, setLoadingDiskInfo] = useState(false);

    const showError = useCallback((message: string) => {
        setError(message);
        setTimeout(() => setError(''), 5000);
    }, []);

    const showSuccess = useCallback((message: string) => {
        setSuccess(message);
        setTimeout(() => setSuccess(''), 3000);
    }, []);

    const loadDatasetState = useCallback(async () => {
        try {
            const data = await apiService.getDatasetState();
            setDatasetState(data);
        } catch (err) {
            showError('Failed to load dataset state');
            console.error(err);
        }
    }, [showError]);

    const loadConfig = useCallback(async () => {
        try {
            const data = await apiService.getConfig();
            console.log("GETTT", data);
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

    const updateConfig = useCallback(async (newConfig: Config) => {
        try {
            setLoading(prev => ({ ...prev, config: true }));
            await apiService.updateConfig(newConfig);
            setConfig(newConfig);
            showSuccess('Configuration updated successfully');
            await loadDatasetState();
        } catch (err) {
            showError('Failed to update configuration');
            console.error(err);
            throw err;
        } finally {
            setLoading(prev => ({ ...prev, config: false }));
        }
    }, [showSuccess, showError, loadDatasetState]);

    const uploadFiles = useCallback(async (files: FileList, type: 'train' | 'pool') => {
        try {
            setLoading(prev => ({ ...prev, [type]: true }));

            const result = await apiService.uploadImages(files, type)

            showSuccess(result.message);
            await loadDatasetState();
        } catch (err) {
            showError(`Failed to upload ${type} files`);
            console.error(err);
        } finally {
            setLoading(prev => ({ ...prev, [type]: false }));
        }
    }, [showError, showSuccess, loadDatasetState]);

    const selectSamples = useCallback(async () => {
        try {
            setLoading(prev => ({ ...prev, select: true }));
            if (datasetState.pool_count < config.budget) {
                showError("Current pool size must greater than budget!");
                return;
            }
            const result = await apiService.selectSamples();
            setSelectedSamples(result.selected_images);
            showSuccess('Sample selection completed');
            await loadDatasetState();
        } catch (err) {
            showError('Failed to select samples');
            console.error(err);
        } finally {
            setLoading(prev => ({ ...prev, select: false }));
        }
    }, [datasetState, config, showSuccess, loadDatasetState, showError]);

    const loadPseudoLabel = useCallback(async (imageIndex: number) => {
        try {
            setLoading(prev => ({ ...prev, pseudo: true }));
            const imagePath = selectedSamples[imageIndex].path;
            const result = await apiService.getPseudoLabel(imagePath);
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
    }, [selectedSamples, showError]);

    const loadAnnotatedSamples = useCallback(async () => {
        try {
            const result = await apiService.getAnnotatedSamples();
            setAnnotatedSamples(result.annotated_samples);
        } catch (err) {
            showError(`Failed to load annotated samples`);
            console.error(err);
        }
    }, [showError]);

    const submitAnnotation = useCallback(async (annotationData: AnnotationData) => {
        if (!pseudoLabel || selectedImageIndex === null) return;

        try {
            setLoading(prev => ({ ...prev, annotate: true }));
            console.log("ANNOTATION DATA", annotationData);
            const response = await apiService.submitAnnotation(annotationData);

            showSuccess(response.message);
            await loadAnnotatedSamples();
            await loadDatasetState();

        } catch (err) {
            showError('Failed to submit annotation');
            console.error(err);
            cancelAnnotation();
            const result = await apiService.getSelectedSamples();
            const newSelectedSamples = result.selected_samples;
            console.log("NEW SELECTED SAMPLES", newSelectedSamples);
            setSelectedSamples(newSelectedSamples);
        } finally {
            setLoading(prev => ({ ...prev, annotate: false }));
            cancelAnnotation();
            const result = await apiService.getSelectedSamples();
            const newSelectedSamples = result.selected_samples;
            console.log("NEW SELECTED SAMPLES", newSelectedSamples);
            setSelectedSamples(newSelectedSamples);
        }
    }, [pseudoLabel, selectedImageIndex, showSuccess, loadAnnotatedSamples, loadDatasetState, showError]);

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

            setSelectedSamples([]);
            setAnnotatedSamples([]);
            setSelectedImageIndex(null);
            setPseudoLabel(null);
            setIsAnnotating(false);
            setTrainFiles(null);
            setPoolFiles(null);

            showSuccess('System reset successfully');
            await loadDatasetState();
        } catch (err) {
            showError('Failed to reset system');
            console.error(err)
        } finally {
            setLoading((prev: any) => ({ ...prev, reset: false }));
        }
    }, [showError, showSuccess, loadDatasetState]);

        const loadDiskInfo = useCallback(async () => {
        setLoadingDiskInfo(true);
        try {
            const info = await apiService.getDiskInfo();
            setDiskInfo(info);
        } catch (err) {
            console.error('Failed to load disk info:', err);
            setError('Failed to load storage information');
        } finally {
            setLoadingDiskInfo(false);
        }
    }, []);

    const syncSystem = useCallback(async () => {
        try {
            setLoading(prev => ({ ...prev, sync: true }));
            await loadConfig();
            await loadDatasetState();
            await loadAnnotatedSamples();
            await loadDiskInfo();
            showSuccess('System reset successfully');
        } catch (error) {
            showError('Failed to load sync system');
            console.error(error);
        } finally {
            setLoading(prev => ({ ...prev, sync: false }));
        }
    }, [loadConfig, loadDatasetState, loadAnnotatedSamples, loadDiskInfo, showSuccess, showError]);

    const startAnnotation = useCallback((index: number) => {
        setSelectedImageIndex(index);
        setIsAnnotating(true);
        loadPseudoLabel(index).then();
    }, [loadPseudoLabel]);

    const cancelAnnotation = () => {
        setIsAnnotating(false);
        setSelectedImageIndex(null);
    };

    useEffect(() => {
        const interval = setInterval(loadDiskInfo, 30000);
        return () => clearInterval(interval);
    }, [loadDiskInfo]);

    useEffect(() => {
        loadDatasetState().then();
        loadConfig().then();
        loadAvailableCheckpoints().then();
        loadAnnotatedSamples().then();
        loadDiskInfo().then();
    }, [loadDatasetState, loadConfig, loadAvailableCheckpoints, loadAnnotatedSamples, loadDiskInfo]);

    return {
        // State
        config,
        datasetState,
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
        diskInfo,
        loadingDiskInfo,

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
        showSuccess,
        loadDiskInfo,
    };
};
