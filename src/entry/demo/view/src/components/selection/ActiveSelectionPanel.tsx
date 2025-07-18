import React from 'react';
import { Button, Badge, Empty, Tooltip } from 'antd';
import {
    PlayCircleOutlined,
    FileImageOutlined,
    LoadingOutlined,
    EditOutlined
} from '@ant-design/icons';
import { Brain } from 'lucide-react';
import type {ActiveLearningState, SelectedSample} from '../../models';

interface ActiveSelectionPanelProps {
    selectedSamples: SelectedSample[];
    status: ActiveLearningState;
    isSelecting: boolean;
    onSelectSamples: () => void;
    onStartAnnotation: (index: number) => void;
}

export const ActiveSelectionPanel: React.FC<ActiveSelectionPanelProps> = ({
                                                                              selectedSamples,
                                                                              status,
                                                                              isSelecting,
                                                                              onSelectSamples,
                                                                              onStartAnnotation,
                                                                          }) => {
    return (
        <div className="bg-white rounded-lg shadow p-6">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-2">
                    <Brain className="h-5 w-5 text-purple-600" />
                    <h2 className="text-lg font-semibold">Active Selection</h2>
                    {selectedSamples.length > 0 && (
                        <Badge
                            count={selectedSamples.length}
                            style={{ backgroundColor: '#722ed1' }}
                        />
                    )}
                </div>

                <Tooltip title={status.pool_count === 0 ? "No pool samples available" : ""}>
                    <Button
                        type="primary"
                        icon={isSelecting ? <LoadingOutlined /> : <PlayCircleOutlined />}
                        loading={isSelecting}
                        disabled={status.pool_count === 0}
                        onClick={onSelectSamples}
                    >
                        {isSelecting ? 'Selecting...' : 'Select Samples'}
                    </Button>
                </Tooltip>
            </div>

            {/* Content */}
            {selectedSamples.length > 0 ? (
                <div>
                    <div className="mb-4">
                        <span className="text-sm font-medium text-gray-700">
                            Selected Samples ({selectedSamples.length})
                        </span>
                    </div>

                    <div className="mt-2 grid grid-cols-4 sm:grid-cols-1 gap-3 max-h-96 overflow-y-auto">
                        {selectedSamples.map((sample, index) => (
                            <div
                                key={index}
                                onClick={() => onStartAnnotation(index)}
                                className="group relative cursor-pointer border border-gray-200 rounded-lg overflow-hidden hover:border-blue-500 transition-all duration-200"
                            >
                                <div className="m-2 aspect-square bg-gray-100 flex items-center justify-center overflow-hidden">
                                    {sample.data ? (
                                        <img
                                            src={`data:image/jpeg;base64,${sample.data}`}
                                            alt={sample.name}
                                            className="w-full h-full object-cover m-2"
                                            onError={(e) => {
                                                e.currentTarget.style.display = 'none';
                                                e.currentTarget.nextElementSibling?.classList.remove('hidden');
                                            }}
                                        />
                                    ) : null}
                                    <FileImageOutlined className="text-2xl text-gray-400 hidden" />
                                </div>

                                {/* Hover overlay */}
                                <div className="absolute inset-0  bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-200 flex items-center justify-center">
                                    <div className="opacity-0 group-hover:opacity-100 bg-opacity-70 text-white px-2 py-1 rounded text-xs flex items-center space-x-1">
                                        <EditOutlined />
                                        <span>Annotate</span>
                                    </div>
                                </div>

                                <Tooltip title={sample.path}>
                                    <div className="absolute bottom-0 left-0 right-0 bg-gray-50 border-t border-gray-200 p-1">
                                        <span className="text-xs text-gray-600 truncate block">
                                            {sample.name}
                                        </span>
                                    </div>
                                </Tooltip>
                            </div>
                        ))}
                    </div>
                </div>
            ) : (
                <Empty
                    image={Empty.PRESENTED_IMAGE_SIMPLE}
                    description="No samples selected yet"
                    className="py-8"
                />
            )}
        </div>
    );
};