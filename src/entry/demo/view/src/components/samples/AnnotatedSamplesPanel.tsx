import React from "react";
import type { AnnotatedSample } from "../../models";
import { Button, Empty, Image, Tooltip, Badge, Space, Tag, Modal, message } from "antd";
import { DownloadOutlined, LoadingOutlined, EyeOutlined, DeleteOutlined, FolderOpenOutlined } from "@ant-design/icons";
import { FileImage } from "lucide-react";

export const AnnotatedSamplesPanel: React.FC<{
    samples: AnnotatedSample[];
    isDownloading: boolean;
    onDownload: () => void;
    onDelete?: (sampleId: string) => void;
}> = ({ samples, isDownloading, onDownload, onDelete }) => {

    const getClassColor = (classId: number) => {
        const colors = {
            1: '#ff4d4f', // Red
            2: '#52c41a', // Green
        };
        return colors[classId as keyof typeof colors] || '#666';
    };

    const formatTimestamp = (timestamp: number) => {
        return new Date(timestamp * 1000).toLocaleString();
    };

    // Helper function to get safe image source
    const getSafeImageSrc = (sample: AnnotatedSample, type: 'background' | 'visual' = 'visual') => {
        let imageData: string | undefined;

        if (type === 'background') {
            imageData = sample.background_image || sample.visual;
        } else {
            imageData = sample.visual;
        }

        // Check if imageData is valid
        if (!imageData || imageData === 'undefined' || imageData === 'null' || imageData.trim() === '') {
            console.warn(`Invalid image data for sample:`, sample);
            return null;
        }

        return `data:image/png;base64,${imageData}`;
    };

    // Helper function to check if sample has valid visual data
    const hasValidVisualData = (sample: AnnotatedSample) => {
        return sample.visual &&
            sample.visual !== 'undefined' &&
            sample.visual !== 'null' &&
            sample.visual.trim() !== '';
    };

    const downloadSingleImage = (sample: AnnotatedSample, type: 'background' | 'visual') => {
        const imageSrc = getSafeImageSrc(sample, type);

        if (!imageSrc) {
            message.error(`No ${type} image data available for this sample`);
            return;
        }

        const link = document.createElement('a');
        link.href = imageSrc;
        link.download = `${sample.case_name || `sample_${sample.index}`}_${type}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };

    const handlePreview = (sample: AnnotatedSample) => {
        const imageSrc = getSafeImageSrc(sample, 'visual');

        if (!imageSrc) {
            message.error('No image data available for preview');
            return;
        }

        Modal.info({
            title: `Annotation Preview - ${sample.case_name || `Sample ${sample.index}`}`,
            width: 800,
            content: (
                <div>
                    <Image
                        src={imageSrc}
                        alt={`Annotation ${sample.case_name || sample.index}`}
                        style={{ width: '100%' }}
                        fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMIAAADDCAYAAADQvc6UAAABRWlDQ1BJQ0MgUHJvZmlsZQAAKJFjYGASSSwoyGFhYGDIzSspCnJ3UoiIjFJgf8LAwSDCIMogwMCcmFxc4BgQ4ANUwgCjUcG3awyMIPqyLsis7PPOq3QdDFcvjV3jOD1boQVTPQrgSkktTgbSf4A4LbmgqISBgTEFyFYuLykAsTuAbJEioKOA7DkgdjqEvQHEToKwj4DVhAQ5A9k3gGyB5IxEoBmML4BsnSQk8XQkNtReEOBxcfXxUQg1Mjc0dyHgXNJBSWpFCYh2zi+oLMpMzyhRcASGUqqCZ16yno6CkYGRAQMDKMwhqj/fAIcloxgHQqxAjIHBEugw5sUIsSQpBobtQPdLciLEVJYzMPBHMDBsayhILEqEO4DxG0txmrERhM29nYGBddr//5/DGRjYNRkY/l7////39v///y4Dmn+LgeHANwDrkl1AuO+pmgAAADhlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAAqACAAQAAAABAAAAwqADAAQAAAABAAAAwwAAAAD9b/HnAAAHlklEQVR4Ae3dP3Ik1RnG4W+FgYxN" // Placeholder image
                    />
                    <div style={{ marginTop: 16 }}>
                        <Space direction="vertical" size="small" style={{ width: '100%' }}>
                            {sample.classes_found && sample.classes_found.length > 0 && (
                                <div>
                                    <span style={{ fontWeight: 'bold' }}>Classes found: </span>
                                    {sample.classes_found.map(classId => (
                                        <Tag key={classId} color={getClassColor(classId)}>
                                            Class {classId}
                                        </Tag>
                                    ))}
                                </div>
                            )}
                            {sample.timestamp && (
                                <div>
                                    <span style={{ fontWeight: 'bold' }}>Created: </span>
                                    {formatTimestamp(sample.timestamp)}
                                </div>
                            )}
                            {sample.dataset_structure && (
                                <div>
                                    <span style={{ fontWeight: 'bold' }}>Saved to: </span>
                                    {sample.dataset_structure.images_folder}{sample.dataset_structure.image_file}
                                </div>
                            )}
                        </Space>
                    </div>
                </div>
            )
        });
    };

    const handleDelete = (sample: AnnotatedSample) => {
        Modal.confirm({
            title: 'Delete Annotated Sample',
            content: `Are you sure you want to delete "${sample.case_name || `Sample ${sample.index}`}"?`,
            onOk: () => {
                if (onDelete) {
                    onDelete(sample.id || sample.case_name || `${sample.index}`);
                    message.success('Annotated sample deleted');
                }
            }
        });
    };

    // Filter out samples with invalid visual data
    const validSamples = samples.filter(hasValidVisualData);

    return (
        <div className="bg-white rounded-lg shadow p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                    <FileImage className="h-5 w-5 text-blue-500" />
                    <h2 className="text-lg font-semibold m-0">
                        Annotated Samples
                    </h2>
                    {validSamples.length > 0 && (
                        <Badge
                            count={validSamples.length}
                            style={{
                                backgroundColor: '#52c41a'
                            }}
                        />
                    )}
                    {samples.length > validSamples.length && (
                        <Tooltip title={`${samples.length - validSamples.length} samples with invalid data hidden`}>
                            <Badge
                                count={samples.length - validSamples.length}
                                style={{
                                    backgroundColor: '#ff4d4f'
                                }}
                            />
                        </Tooltip>
                    )}
                </div>

                <Space>
                    {validSamples.length > 0 && (
                        <Tooltip title="Dataset structure: images/ and labels/ folders">
                            <Button
                                icon={<FolderOpenOutlined />}
                                type="dashed"
                                size="small"
                            >
                                Structure
                            </Button>
                        </Tooltip>
                    )}
                    <Button
                        type="primary"
                        icon={isDownloading ? <LoadingOutlined /> : <DownloadOutlined />}
                        loading={isDownloading}
                        disabled={validSamples.length === 0}
                        onClick={onDownload}
                        className="!bg-green-500 !border-green-500 hover:!bg-green-500/80 hover:!border-green-500/80"
                    >
                        {isDownloading ? 'Downloading...' : 'Download Dataset'}
                    </Button>
                </Space>
            </div>

            {/* Content */}
            {validSamples.length > 0 ? (
                <div className="max-h-96 overflow-y-auto pr-1">
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {validSamples.map((sample, index) => {
                            const imageSrc = getSafeImageSrc(sample, 'visual');

                            return (
                                <div
                                    key={sample.id || sample.case_name || index}
                                    className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow duration-200 group"
                                >
                                    <div className="h-40 bg-gray-50 flex items-center justify-center overflow-hidden relative">
                                        {imageSrc ? (
                                            <Image
                                                src={imageSrc}
                                                alt={`Annotated ${sample.case_name || index}`}
                                                className="w-full h-full object-contain"
                                                preview={false}
                                                fallback="/api/placeholder/150/150"
                                                onError={(e) => {
                                                    console.error('Image failed to load:', sample);
                                                    e.currentTarget.style.display = 'none';
                                                }}
                                            />
                                        ) : (
                                            <div className="text-gray-400 text-center p-4">
                                                <FileImage className="h-8 w-8 mx-auto mb-2" />
                                                <span className="text-sm">No image data</span>
                                            </div>
                                        )}

                                        {/* Hover overlay with actions */}
                                        <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all duration-200 flex items-center justify-center opacity-0 group-hover:opacity-100">
                                            <Space>
                                                <Tooltip title="Preview">
                                                    <Button
                                                        type="primary"
                                                        icon={<EyeOutlined />}
                                                        size="small"
                                                        onClick={() => handlePreview(sample)}
                                                        disabled={!imageSrc}
                                                    />
                                                </Tooltip>
                                                <Tooltip title="Download Original">
                                                    <Button
                                                        icon={<DownloadOutlined />}
                                                        size="small"
                                                        onClick={() => downloadSingleImage(sample, 'background')}
                                                        disabled={!getSafeImageSrc(sample, 'background')}
                                                    >
                                                        IMG
                                                    </Button>
                                                </Tooltip>
                                                <Tooltip title="Download Visual">
                                                    <Button
                                                        icon={<DownloadOutlined />}
                                                        size="small"
                                                        onClick={() => downloadSingleImage(sample, 'visual')}
                                                        disabled={!imageSrc}
                                                    >
                                                        VIS
                                                    </Button>
                                                </Tooltip>
                                                {onDelete && (
                                                    <Tooltip title="Delete">
                                                        <Button
                                                            danger
                                                            icon={<DeleteOutlined />}
                                                            size="small"
                                                            onClick={() => handleDelete(sample)}
                                                        />
                                                    </Tooltip>
                                                )}
                                            </Space>
                                        </div>
                                    </div>

                                    <div className="p-3">
                                        <div className="mb-2">
                                            <Tooltip title={sample.path}>
                                                <span className="text-sm font-medium text-gray-800 block truncate">
                                                    {sample.case_name || sample.path?.split('/').pop() || `Sample ${sample.index}`}
                                                </span>
                                            </Tooltip>
                                        </div>

                                        {/* Classes */}
                                        {sample.classes_found && sample.classes_found.length > 0 && (
                                            <div className="mb-2">
                                                <Space size={[4, 4]} wrap>
                                                    {sample.classes_found.map(classId => (
                                                        <Tag
                                                            key={classId}
                                                            color={getClassColor(classId)}
                                                            size="small"
                                                        >
                                                            {classId}
                                                        </Tag>
                                                    ))}
                                                </Space>
                                            </div>
                                        )}

                                        {/* Timestamp */}
                                        {sample.timestamp && (
                                            <div className="text-xs text-gray-400">
                                                {formatTimestamp(sample.timestamp)}
                                            </div>
                                        )}

                                        {/* Dataset structure info */}
                                        {sample.dataset_structure && (
                                            <Tooltip title={`Saved to: ${sample.dataset_structure.images_folder}${sample.dataset_structure.image_file}`}>
                                                <div className="text-xs text-blue-600 mt-1 truncate">
                                                    📁 {sample.dataset_structure.image_file}
                                                </div>
                                            </Tooltip>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            ) : (
                <div>
                    <Empty
                        image={Empty.PRESENTED_IMAGE_SIMPLE}
                        description="No valid annotated samples yet"
                        className="py-8"
                    />
                    {samples.length > 0 && validSamples.length === 0 && (
                        <div className="text-center text-red-500 text-sm mt-2">
                            {samples.length} sample(s) found but all have invalid image data
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};