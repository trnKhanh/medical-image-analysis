// import React from 'react';
// import { Brain, Play, ImageIcon } from 'lucide-react';
// import type {SystemStatus} from '../../models';
//
// interface ActiveSelectionPanelProps {
//     selectedSamples: string[];
//     status: SystemStatus;
//     isSelecting: boolean;
//     onSelectSamples: () => void;
//     onStartAnnotation: (index: number) => void;
// }
//
// export const ActiveSelectionPanel: React.FC<ActiveSelectionPanelProps> = ({
//                                                                               selectedSamples,
//                                                                               status,
//                                                                               isSelecting,
//                                                                               onSelectSamples,
//                                                                               onStartAnnotation
//                                                                           }) => {
//     return (
//         <div className="bg-white rounded-lg shadow p-6">
//             <div className="flex items-center justify-between mb-4">
//                 <div className="flex items-center space-x-2">
//                     <Brain className="h-5 w-5 text-gray-600" />
//                     <h2 className="text-lg font-semibold">Active Selection</h2>
//                 </div>
//                 <button
//                     onClick={onSelectSamples}
//                     disabled={isSelecting || status.pool_set_size === 0}
//                     className="flex items-center space-x-2 px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50"
//                 >
//                     <Play className="h-4 w-4" />
//                     <span>{isSelecting ? 'Selecting...' : 'Select Samples'}</span>
//                 </button>
//             </div>
//
//             {selectedSamples.length > 0 && (
//                 <div>
//                     <h3 className="text-sm font-medium text-gray-700 mb-2">
//                         Selected Samples ({selectedSamples.length})
//                     </h3>
//                     <div className="grid grid-cols-2 gap-2 max-h-96 overflow-y-auto">
//                         {selectedSamples.map((sample, index) => (
//                             <div
//                                 key={index}
//                                 className="relative group cursor-pointer border rounded-lg overflow-hidden hover:border-blue-500"
//                                 onClick={() => onStartAnnotation(index)}
//                             >
//                                 <div className="aspect-square bg-gray-100 flex items-center justify-center">
//                                     <ImageIcon className="h-8 w-8 text-gray-400" />
//                                 </div>
//                                 <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-200 flex items-center justify-center">
//                                     <span className="text-white text-sm font-medium opacity-0 group-hover:opacity-100">
//                                         Annotate
//                                     </span>
//                                 </div>
//                                 <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-1 truncate">
//                                     {sample.split('/').pop()}
//                                 </div>
//                             </div>
//                         ))}
//                     </div>
//                 </div>
//             )}
//         </div>
//     );
// };
import React from 'react';
import { Button, Badge, Empty, Tooltip } from 'antd';
import {
    PlayCircleOutlined,
    FileImageOutlined,
    LoadingOutlined,
    EditOutlined
} from '@ant-design/icons';
import { Brain } from 'lucide-react';
import type { SystemStatus } from '../../models';

interface ActiveSelectionPanelProps {
    selectedSamples: string[];
    status: SystemStatus;
    isSelecting: boolean;
    onSelectSamples: () => void;
    onStartAnnotation: (index: number) => void;
}

export const ActiveSelectionPanel: React.FC<ActiveSelectionPanelProps> = ({
                                                                              selectedSamples,
                                                                              status,
                                                                              isSelecting,
                                                                              onSelectSamples,
                                                                              onStartAnnotation
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

                <Tooltip title={status.pool_set_size === 0 ? "No pool samples available" : ""}>
                    <Button
                        type="primary"
                        icon={isSelecting ? <LoadingOutlined /> : <PlayCircleOutlined />}
                        loading={isSelecting}
                        disabled={status.pool_set_size === 0}
                        onClick={onSelectSamples}
                        className="bg-purple-600 border-purple-600 hover:bg-purple-700"
                    >
                        {isSelecting ? 'Selecting...' : 'Select Samples'}
                    </Button>
                </Tooltip>
            </div>

            {/* Content */}
            {selectedSamples.length > 0 ? (
                <div>
                    <div className="mb-3">
                        <span className="text-sm font-medium text-gray-700">
                            Selected Samples ({selectedSamples.length})
                        </span>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 max-h-96 overflow-y-auto">
                        {selectedSamples.map((sample, index) => (
                            <div
                                key={index}
                                onClick={() => onStartAnnotation(index)}
                                className="group relative cursor-pointer border border-gray-200 rounded-lg overflow-hidden hover:border-blue-500 transition-all duration-200"
                            >
                                <div className="aspect-square bg-gray-100 flex items-center justify-center">
                                    <FileImageOutlined className="text-2xl text-gray-400" />
                                </div>

                                {/* Hover overlay */}
                                <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-20 transition-all duration-200 flex items-center justify-center">
                                    <div className="opacity-0 group-hover:opacity-100 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-xs flex items-center space-x-1">
                                        <EditOutlined />
                                        <span>Annotate</span>
                                    </div>
                                </div>

                                <Tooltip title={sample}>
                                    <div className="absolute bottom-0 left-0 right-0 bg-gray-50 border-t border-gray-200 p-1">
                                        <span className="text-xs text-gray-600 truncate block">
                                            {sample.split('/').pop()}
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