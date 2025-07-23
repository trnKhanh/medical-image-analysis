import React from 'react';
import { Header } from './components/layout/Header';
import { Notification } from './components/Notification';
import { ConfigurationPanel } from './components/configuration/ConfigurationPanel';
import { FileUploadPanel } from './components/upload/FileUploadPanel';
import { ActiveSelectionPanel } from './components/selection/ActiveSelectionPanel';
import { AnnotationEditor } from './components/annotation/AnnotationEditor';
import { AnnotatedSamplesPanel } from './components/samples/AnnotatedSamplesPanel';
import { useApp } from './hooks/useApp';
import './styles/App.css';
import {Modal} from "antd";

const App: React.FC = () => {
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

        // Actions
        setTrainFiles,
        setPoolFiles,
        setBrushColor,
        uploadFiles,
        selectSamples,
        submitAnnotation,
        cancelAnnotation,
        downloadDataset,
        resetSystem,
        syncSystem,
        startAnnotation,
        loadAvailableCheckpoints
    } = useApp();

    return (
        <div className="min-h-screen bg-gray-50">
            <Header
                status={status}
                onReset={resetSystem}
                isResetting={loading.reset}
                isSyncing={loading.sync}
                onSync={syncSystem}
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

            {/* Main Content */}
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
                        />
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
                                    imageIndex={selectedImageIndex}
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

export default App;
