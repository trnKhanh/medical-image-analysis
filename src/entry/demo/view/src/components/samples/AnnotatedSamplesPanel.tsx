import React from "react";
import { Button, Empty, Image, Badge, Space } from "antd";
import { DownloadOutlined, LoadingOutlined } from "@ant-design/icons";
import { FileImage } from "lucide-react";
import type {AnnotatedSample} from "../../models";

export const AnnotatedSamplesPanel: React.FC<{
    samples: AnnotatedSample[];
    isDownloading: boolean;
    onDownload: () => void;
    onDelete?: (sampleId: string) => void;
}> = ({ samples, isDownloading, onDownload }) => {

    return (
        <div className="bg-white rounded-lg shadow p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                    <FileImage className="h-5 w-5 text-blue-500" />
                    <h2 className="text-lg font-semibold m-0">
                        Annotated Samples
                    </h2>
                    {samples.length > 0 && (
                        <Badge
                            count={samples.length}
                            style={{
                                backgroundColor: '#52c41a'
                            }}
                        />
                    )}
                </div>

                <Space>
                    <Button
                        type="primary"
                        icon={isDownloading ? <LoadingOutlined /> : <DownloadOutlined />}
                        loading={isDownloading}
                        disabled={samples.length === 0}
                        onClick={onDownload}
                        className="!bg-green-500 !border-green-500 hover:!bg-green-500/80 hover:!border-green-500/80"
                    >
                        {isDownloading ? 'Downloading...' : 'Download Dataset'}
                    </Button>
                </Space>
            </div>

            {/* Content - Responsive Grid */}
            {samples.length > 0 ? (
                <div className="max-h-96 overflow-y-auto pr-1">
                    {/* Responsive Grid: 4 cols on xl+, 3 cols on lg, 2 cols on md and below */}
                    <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
                        {samples.map((sample, index) => (
                            <div
                                key={index}
                                className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow duration-200"
                            >
                                <div className="aspect-square bg-gray-50 flex items-center justify-center overflow-hidden">
                                    {sample ? (
                                        <Image
                                            src={`data:image/png;base64,${sample}`}
                                            alt={`Sample ${index + 1}`}
                                            className="w-full h-full object-cover"
                                            preview={true}
                                        />
                                    ) : (
                                        <div className="text-gray-400 text-center p-4">
                                            <FileImage className="h-8 w-8 mx-auto mb-2" />
                                            <span className="text-sm">No image</span>
                                        </div>
                                    )}
                                </div>

                                <div className="p-3">
                                    <span className="text-sm font-medium text-gray-800 block truncate">
                                        Sample {index + 1}
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            ) : (
                <Empty
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description="No annotated samples yet"
                    className="py-8"
                />
            )}
        </div>
    );
};