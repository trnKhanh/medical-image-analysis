// import type {AnnotatedSample} from "../../models";
// import { Download } from "lucide-react";
//
// export const AnnotatedSamplesPanel: React.FC<{
//     samples: AnnotatedSample[];
//     isDownloading: boolean;
//     onDownload: () => void;
// }> = ({ samples, isDownloading, onDownload }) => {
//     return (
//         <div className="bg-white rounded-lg shadow p-6">
//             <div className="flex items-center justify-between mb-4">
//                 <h2 className="text-lg font-semibold">Annotated Samples</h2>
//                 <button
//                     onClick={onDownload}
//                     disabled={isDownloading || samples.length === 0}
//                     className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
//                 >
//                     <Download className="h-4 w-4" />
//                     <span>{isDownloading ? 'Downloading...' : 'Download Dataset'}</span>
//                 </button>
//             </div>
//
//             {samples.length > 0 ? (
//                 <div className="grid grid-cols-1 gap-4 max-h-96 overflow-y-auto">
//                     {samples.map((sample, index) => (
//                         <div key={index} className="border rounded-lg overflow-hidden">
//                             <div className="aspect-square bg-gray-100 flex items-center justify-center">
//                                 <img
//                                     src={`data:image/png;base64,${sample.visual}`}
//                                     alt={`Annotated ${index}`}
//                                     className="w-full h-full object-contain"
//                                 />
//                             </div>
//                             <div className="p-2 bg-gray-50">
//                                 <span className="text-xs text-gray-600 truncate block">
//                                     {sample.path.split('/').pop()}
//                                 </span>
//                             </div>
//                         </div>
//                     ))}
//                 </div>
//             ) : (
//                 <div className="text-center py-8 text-gray-500">
//                     No annotated samples yet
//                 </div>
//             )}
//         </div>
//     );
// };
import React from "react";
import type { AnnotatedSample } from "../../models";
import { Button, Empty, Image, Tooltip, Badge } from "antd";
import { DownloadOutlined, LoadingOutlined } from "@ant-design/icons";
import { FileImage } from "lucide-react";

export const AnnotatedSamplesPanel: React.FC<{
    samples: AnnotatedSample[];
    isDownloading: boolean;
    onDownload: () => void;
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
            </div>

            {/* Content */}
            {samples.length > 0 ? (
                <div className="max-h-96 overflow-y-auto pr-1">
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {samples.map((sample, index) => (
                            <div
                                key={index}
                                className="border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow duration-200"
                            >
                                <div className="h-40 bg-gray-50 flex items-center justify-center overflow-hidden">
                                    <Image
                                        src={`data:image/png;base64,${sample.visual}`}
                                        alt={`Annotated ${index}`}
                                        className="w-full h-full object-contain"
                                        preview={{
                                            mask: (
                                                <div className="bg-black bg-opacity-50 text-white px-2 py-1 rounded text-xs">
                                                    Preview
                                                </div>
                                            )
                                        }}
                                    />
                                </div>
                                <div className="p-2">
                                    <Tooltip title={sample.path}>
                                        <span className="text-xs text-gray-500 block truncate">
                                            {sample.path.split('/').pop()}
                                        </span>
                                    </Tooltip>
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