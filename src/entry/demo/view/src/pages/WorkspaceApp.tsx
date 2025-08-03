import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, Tag } from 'antd';
import { HomeOutlined, CopyOutlined } from '@ant-design/icons';
import { Header } from '../components/layout/Header';
import { Notification } from '../components/Notification';
import { ConfigurationPanel } from '../components/configuration/ConfigurationPanel';
import { FileUploadPanel } from '../components/upload/FileUploadPanel';
import { ActiveSelectionPanel } from '../components/selection/ActiveSelectionPanel';
import { AnnotationEditor } from '../components/annotation/AnnotationEditor';
import { AnnotatedSamplesPanel } from '../components/samples/AnnotatedSamplesPanel';
import { useApp } from '../hooks/useApp';
import {apiService} from '../services/api';
import { Modal, message } from 'antd';
import '../styles/App.css';
import {DiskUsageCard} from "../components/DiskUsage.tsx";

interface WorkspaceAppProps {
    workspaceId: string;
}

export const WorkspaceApp: React.FC<WorkspaceAppProps> = ({ workspaceId }) => {
    const navigate = useNavigate();

    const {
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
        diskInfo,
        loadingDiskInfo,

        // Actions
        setTrainFiles,
        setPoolFiles,
        setBrushColor,
        uploadFiles,
        updateConfig,
        selectSamples,
        submitAnnotation,
        cancelAnnotation,
        downloadDataset,
        resetSystem,
        syncSystem,
        startAnnotation,
        loadAvailableCheckpoints,
        loadDiskInfo
    } = useApp();

    useEffect(() => {
        apiService.setWorkspace(workspaceId);

        syncSystem().then();
    }, [syncSystem, workspaceId]);

    const handleBackToHome = () => {
        navigate('/');
    };

    const copyWorkspaceUrl = () => {
        const url = window.location.href;
        navigator.clipboard.writeText(url);
        message.success('Workspace URL copied to clipboard!');
    };

    return (

        <div className="min-h-screen bg-gray-50">
            <Header
                status={status}
                onReset={resetSystem}
                isResetting={loading.reset}
                isSyncing={loading.sync}
                onSync={syncSystem}
                extraActions={
                    <div className="flex items-center space-x-2">
                        <Button
                            icon={<CopyOutlined />}
                            onClick={copyWorkspaceUrl}
                            type="text"
                            size="small"
                        >
                            Copy URL
                        </Button>
                        <Button
                            icon={<HomeOutlined />}
                            onClick={handleBackToHome}
                            type="text"
                        >
                            New Workspace
                        </Button>
                    </div>
                }
            />

            {error && (
                <Notification
                    type="error"
                    message={error}
                />
            )}

            {success && (
                <Notification
                    type="success"
                    message={success}
                />
            )}

            {/* Workspace indicator bar */}
            <div className="bg-blue-50 border-b border-blue-200 px-4 py-3">
                <div className="max-w-7xl mx-auto flex items-center justify-between mt-3">
                    <div className="flex items-center space-x-3">
                        <span className="text-blue-700 font-medium">Current Workspace:</span>
                        <Tag color="blue" className="font-mono text-sm px-3 py-1">
                            {workspaceId}
                        </Tag>
                    </div>
                </div>
            </div>

            {/* Main Content - Your existing layout */}
            <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
                <div className="grid grid-cols-1 lg:custom-grid-cols-2 gap-6">
                    <div className="space-y-6">
                        <FileUploadPanel
                            trainFiles={trainFiles}
                            poolFiles={poolFiles}
                            loading={{ train: loading.train, pool: loading.pool }}
                            onTrainFilesChange={setTrainFiles}
                            onPoolFilesChange={setPoolFiles}
                            onUploadTrain={async () => {
                                if (trainFiles)
                                    await uploadFiles(trainFiles, 'train');
                                setTrainFiles(null);
                            }}
                            onUploadPool={async () => {
                                if (poolFiles)
                                    await uploadFiles(poolFiles, 'pool');
                                setPoolFiles(null)
                            }}
                        />
                    </div>
                    <div className="space-y-6">
                        <ConfigurationPanel
                            config={config}
                            checkpoints={availableCheckpoints}
                            loadingCheckpoints={loadingCheckpoints}
                            onRefreshCheckpoints={loadAvailableCheckpoints}
                            onUpdateConfig={updateConfig}
                        />
                        <DiskUsageCard diskInfo={diskInfo} loading={loadingDiskInfo} onRefresh={loadDiskInfo}/>
                    </div>
                </div>

                <div className="grid grid-cols-1 gap-6">
                    <div className="space-y-6">
                        <ActiveSelectionPanel
                            selectedSamples={selectedSamples}
                            status={status}
                            isSelecting={loading.select}
                            onSelectSamples={selectSamples}
                            onStartAnnotation={startAnnotation}
                        />

                        <Modal
                            open={isAnnotating && !!pseudoLabel && selectedImageIndex !== null}
                            title="Annotation Editor"
                            onCancel={cancelAnnotation}
                            footer={null}
                            width="auto"
                            style={{ top: 24 }}
                        >
                            {isAnnotating && pseudoLabel && selectedImageIndex !== null && (
                                <AnnotationEditor
                                    pseudoLabel={pseudoLabel}
                                    selectedImageContent={selectedSamples[selectedImageIndex].data}
                                    brushColor={brushColor}
                                    isSubmitting={loading.annotate}
                                    imagePath={selectedSamples[selectedImageIndex].path}
                                    onBrushColorChange={setBrushColor}
                                    onSubmitAnnotation={submitAnnotation}
                                />
                            )}
                        </Modal>
                    </div>

                    <div className="space-y-6">
                        <AnnotatedSamplesPanel
                            samples={annotatedSamples}
                            isDownloading={loading.download}
                            onDownload={downloadDataset}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
};
