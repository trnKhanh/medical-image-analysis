// import type {PseudoLabel} from "../../models";
// import React from "react";
// import {CheckCircle} from "lucide-react";
//
//
// export const AnnotationEditor: React.FC<{
//     pseudoLabel: PseudoLabel;
//     selectedImagePath: string;
//     brushColor: string;
//     isSubmitting: boolean;
//     onBrushColorChange: (color: string) => void;
//     onSubmitAnnotation: () => void;
// }> = ({ pseudoLabel, selectedImagePath, brushColor, isSubmitting, onBrushColorChange, onSubmitAnnotation }) => {
//
//     // Convert pseudoLabel background to displayable image
//     const backgroundImageSrc = React.useMemo(() => {
//         if (!pseudoLabel?.background) return null;
//
//         const canvas = document.createElement('canvas');
//         const ctx = canvas.getContext('2d');
//         if (!ctx) return null;
//
//         const height = pseudoLabel.background.length;
//         const width = pseudoLabel.background[0].length;
//
//         canvas.width = width;
//         canvas.height = height;
//
//         const imageData = ctx.createImageData(width, height);
//
//         for (let y = 0; y < height; y++) {
//             for (let x = 0; x < width; x++) {
//                 const pixelIndex = (y * width + x) * 4;
//                 const pixel = pseudoLabel.background[y][x];
//
//                 // Handle different pixel formats
//                 if (Array.isArray(pixel)) {
//                     imageData.data[pixelIndex] = pixel[0] || 0;     // R
//                     imageData.data[pixelIndex + 1] = pixel[1] || 0; // G
//                     imageData.data[pixelIndex + 2] = pixel[2] || 0; // B
//                     imageData.data[pixelIndex + 3] = pixel[3] !== undefined ? pixel[3] : 255; // A
//                 } else {
//                     // Grayscale
//                     imageData.data[pixelIndex] = pixel;
//                     imageData.data[pixelIndex + 1] = pixel;
//                     imageData.data[pixelIndex + 2] = pixel;
//                     imageData.data[pixelIndex + 3] = 255;
//                 }
//             }
//         }
//
//         ctx.putImageData(imageData, 0, 0);
//         return canvas.toDataURL();
//     }, [pseudoLabel]);
//
//     // Convert layers to overlay images
//     const layerOverlays = React.useMemo(() => {
//         if (!pseudoLabel?.layers || !Array.isArray(pseudoLabel.layers)) return [];
//
//         return pseudoLabel.layers.map((layer) => {
//             const canvas = document.createElement('canvas');
//             const ctx = canvas.getContext('2d');
//             if (!ctx || !layer) return null;
//
//             const height = layer.length;
//             const width = layer[0]?.length || 0;
//
//             canvas.width = width;
//             canvas.height = height;
//
//             const imageData = ctx.createImageData(width, height);
//
//             for (let y = 0; y < height; y++) {
//                 for (let x = 0; x < width; x++) {
//                     const pixelIndex = (y * width + x) * 4;
//                     const pixel = layer[y][x];
//
//                     if (Array.isArray(pixel)) {
//                         imageData.data[pixelIndex] = pixel[0] || 0;
//                         imageData.data[pixelIndex + 1] = pixel[1] || 0;
//                         imageData.data[pixelIndex + 2] = pixel[2] || 0;
//                         imageData.data[pixelIndex + 3] = pixel[3] || 0;
//                     } else {
//                         imageData.data[pixelIndex] = pixel;
//                         imageData.data[pixelIndex + 1] = pixel;
//                         imageData.data[pixelIndex + 2] = pixel;
//                         imageData.data[pixelIndex + 3] = pixel > 0 ? 128 : 0; // Semi-transparent overlay
//                     }
//                 }
//             }
//
//             ctx.putImageData(imageData, 0, 0);
//             return canvas.toDataURL();
//         }).filter(Boolean);
//     }, [pseudoLabel]);
//
//     return (
//         <div className="bg-white rounded-lg shadow p-6">
//             <div className="flex items-center justify-between mb-4">
//                 <h3 className="text-lg font-semibold">Annotation Editor</h3>
//                 <div className="flex items-center space-x-2">
//                     <div className="flex items-center space-x-2">
//                         <span className="text-sm text-gray-600">Brush:</span>
//                         <button
//                             onClick={() => onBrushColorChange('#ff0000')}
//                             className={`w-6 h-6 rounded bg-red-500 border-2 ${
//                                 brushColor === '#ff0000' ? 'border-gray-800' : 'border-gray-300'
//                             }`}
//                             title="Red brush"
//                         />
//                         <button
//                             onClick={() => onBrushColorChange('#00ff00')}
//                             className={`w-6 h-6 rounded bg-green-500 border-2 ${
//                                 brushColor === '#00ff00' ? 'border-gray-800' : 'border-gray-300'
//                             }`}
//                             title="Green brush"
//                         />
//                         <button
//                             onClick={() => onBrushColorChange('#0000ff')}
//                             className={`w-6 h-6 rounded bg-blue-500 border-2 ${
//                                 brushColor === '#0000ff' ? 'border-gray-800' : 'border-gray-300'
//                             }`}
//                             title="Blue brush"
//                         />
//                         <button
//                             onClick={() => onBrushColorChange('#ffffff')}
//                             className={`w-6 h-6 rounded bg-white border-2 ${
//                                 brushColor === '#ffffff' ? 'border-gray-800' : 'border-gray-300'
//                             }`}
//                             title="Eraser"
//                         />
//                     </div>
//                     <button
//                         onClick={onSubmitAnnotation}
//                         disabled={isSubmitting}
//                         className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
//                     >
//                         <CheckCircle className="h-4 w-4" />
//                         <span>{isSubmitting ? 'Submitting...' : 'Accept'}</span>
//                     </button>
//                 </div>
//             </div>
//
//             <div className="border rounded-lg p-4 bg-gray-50">
//                 <div className="text-sm text-gray-600 mb-2">
//                     Image: {selectedImagePath?.split('/').pop()}
//                 </div>
//
//                 {/* Image Display with Pseudo Label */}
//                 <div className="relative w-full h-64 bg-gray-200 rounded overflow-hidden">
//                     {backgroundImageSrc ? (
//                         <div className="relative w-full h-full">
//                             {/* Background Image */}
//                             <img
//                                 src={backgroundImageSrc}
//                                 alt="Background"
//                                 className="absolute inset-0 w-full h-full object-contain"
//                             />
//
//                             {/* Layer Overlays */}
//                             {layerOverlays.map((layerSrc, index) => (
//                                 layerSrc && (
//                                     <img
//                                         key={index}
//                                         src={layerSrc}
//                                         alt={`Layer ${index}`}
//                                         className="absolute inset-0 w-full h-full object-contain opacity-60"
//                                         style={{ mixBlendMode: 'multiply' }}
//                                     />
//                                 )
//                             ))}
//
//                             {/* Canvas for drawing would go here */}
//                             <div className="absolute inset-0 w-full h-full flex items-center justify-center pointer-events-none">
//                                 <span className="text-white bg-black bg-opacity-50 px-2 py-1 rounded text-sm">
//                                     Interactive annotation canvas overlay
//                                 </span>
//                             </div>
//                         </div>
//                     ) : (
//                         <div className="w-full h-full flex items-center justify-center">
//                             <span className="text-gray-500">Loading pseudo label...</span>
//                         </div>
//                     )}
//                 </div>
//
//                 {/* Pseudo Label Info */}
//                 {pseudoLabel && (
//                     <div className="mt-2 text-xs text-gray-500">
//                         <div>Background: {pseudoLabel.background?.length || 0} × {pseudoLabel.background?.[0]?.length || 0}</div>
//                         <div>Layers: {pseudoLabel.layers?.length || 0}</div>
//                         <div>Source: {pseudoLabel.image_path?.split('/').pop() || 'Unknown'}</div>
//                     </div>
//                 )}
//             </div>
//         </div>
//     );
// };

import type { PseudoLabel } from "../../models";
import React from "react";
import { Card, Button, Space, Typography, Tooltip, Tag, Spin, Divider } from "antd";
import { CheckOutlined, BgColorsOutlined, LoadingOutlined } from "@ant-design/icons";

const { Title, Text } = Typography;

export const AnnotationEditor: React.FC<{
    pseudoLabel: PseudoLabel;
    selectedImagePath: string;
    brushColor: string;
    isSubmitting: boolean;
    onBrushColorChange: (color: string) => void;
    onSubmitAnnotation: () => void;
}> = ({ pseudoLabel, selectedImagePath, brushColor, isSubmitting, onBrushColorChange, onSubmitAnnotation }) => {

    // Convert pseudoLabel background to displayable image
    const backgroundImageSrc = React.useMemo(() => {
        if (!pseudoLabel?.background) return null;

        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        if (!ctx) return null;

        const height = pseudoLabel.background.length;
        const width = pseudoLabel.background[0].length;

        canvas.width = width;
        canvas.height = height;

        const imageData = ctx.createImageData(width, height);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const pixelIndex = (y * width + x) * 4;
                const pixel = pseudoLabel.background[y][x];

                // Handle different pixel formats
                if (Array.isArray(pixel)) {
                    imageData.data[pixelIndex] = pixel[0] || 0;     // R
                    imageData.data[pixelIndex + 1] = pixel[1] || 0; // G
                    imageData.data[pixelIndex + 2] = pixel[2] || 0; // B
                    imageData.data[pixelIndex + 3] = pixel[3] !== undefined ? pixel[3] : 255; // A
                } else {
                    // Grayscale
                    imageData.data[pixelIndex] = pixel;
                    imageData.data[pixelIndex + 1] = pixel;
                    imageData.data[pixelIndex + 2] = pixel;
                    imageData.data[pixelIndex + 3] = 255;
                }
            }
        }

        ctx.putImageData(imageData, 0, 0);
        return canvas.toDataURL();
    }, [pseudoLabel]);

    // Convert layers to overlay images
    const layerOverlays = React.useMemo(() => {
        if (!pseudoLabel?.layers || !Array.isArray(pseudoLabel.layers)) return [];

        return pseudoLabel.layers.map((layer) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            if (!ctx || !layer) return null;

            const height = layer.length;
            const width = layer[0]?.length || 0;

            canvas.width = width;
            canvas.height = height;

            const imageData = ctx.createImageData(width, height);

            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    const pixelIndex = (y * width + x) * 4;
                    const pixel = layer[y][x];

                    if (Array.isArray(pixel)) {
                        imageData.data[pixelIndex] = pixel[0] || 0;
                        imageData.data[pixelIndex + 1] = pixel[1] || 0;
                        imageData.data[pixelIndex + 2] = pixel[2] || 0;
                        imageData.data[pixelIndex + 3] = pixel[3] || 0;
                    } else {
                        imageData.data[pixelIndex] = pixel;
                        imageData.data[pixelIndex + 1] = pixel;
                        imageData.data[pixelIndex + 2] = pixel;
                        imageData.data[pixelIndex + 3] = pixel > 0 ? 128 : 0; // Semi-transparent overlay
                    }
                }
            }

            ctx.putImageData(imageData, 0, 0);
            return canvas.toDataURL();
        }).filter(Boolean);
    }, [pseudoLabel]);

    const brushOptions = [
        { color: '#ff0000', name: 'Red', bgColor: '#ff4d4f' },
        { color: '#00ff00', name: 'Green', bgColor: '#52c41a' },
        { color: '#0000ff', name: 'Blue', bgColor: '#1890ff' },
        { color: '#ffffff', name: 'Eraser', bgColor: '#ffffff', textColor: '#000' }
    ];

    return (
        <Card
            size="default"
            style={{
                borderRadius: 8,
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
            }}
        >
            {/* Header */}
            <div style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
                    <Title level={4} style={{ margin: 0 }}>
                        Annotation Editor
                    </Title>

                    <Space size="middle">
                        {/* Brush Color Selector */}
                        <Space>
                            <BgColorsOutlined style={{ color: '#666' }} />
                            <Text type="secondary" style={{ fontSize: 12 }}>Brush:</Text>
                            <Space size="small">
                                {brushOptions.map((brush) => (
                                    <Tooltip key={brush.color} title={brush.name}>
                                        <Button
                                            size="small"
                                            shape="circle"
                                            onClick={() => onBrushColorChange(brush.color)}
                                            style={{
                                                backgroundColor: brush.bgColor,
                                                borderColor: brushColor === brush.color ? '#333' : '#d9d9d9',
                                                borderWidth: brushColor === brush.color ? 2 : 1,
                                                color: brush.textColor || '#fff',
                                                width: 24,
                                                height: 24,
                                                minWidth: 24
                                            }}
                                        />
                                    </Tooltip>
                                ))}
                            </Space>
                        </Space>

                        {/* Accept Button */}
                        <Button
                            type="primary"
                            icon={isSubmitting ? <LoadingOutlined /> : <CheckOutlined />}
                            loading={isSubmitting}
                            onClick={onSubmitAnnotation}
                            size="middle"
                        >
                            {isSubmitting ? 'Submitting...' : 'Accept'}
                        </Button>
                    </Space>
                </div>

                <Divider style={{ margin: 0 }} />
            </div>

            {/* Content */}
            <div style={{
                padding: 16,
                backgroundColor: '#fafafa',
                borderRadius: 6,
                border: '1px solid #f0f0f0'
            }}>
                {/* Image Path */}
                <div style={{ marginBottom: 12 }}>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                        Image: <Text code>{selectedImagePath?.split('/').pop()}</Text>
                    </Text>
                </div>

                {/* Image Display with Pseudo Label */}
                <div style={{
                    position: 'relative',
                    width: '100%',
                    height: 320,
                    backgroundColor: '#f5f5f5',
                    borderRadius: 6,
                    overflow: 'hidden',
                    border: '1px solid #e8e8e8'
                }}>
                    {backgroundImageSrc ? (
                        <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                            {/* Background Image */}
                            <img
                                src={backgroundImageSrc}
                                alt="Background"
                                style={{
                                    position: 'absolute',
                                    top: 0,
                                    left: 0,
                                    width: '100%',
                                    height: '100%',
                                    objectFit: 'contain'
                                }}
                            />

                            {/* Layer Overlays */}
                            {layerOverlays.map((layerSrc, index) => (
                                layerSrc && (
                                    <img
                                        key={index}
                                        src={layerSrc}
                                        alt={`Layer ${index}`}
                                        style={{
                                            position: 'absolute',
                                            top: 0,
                                            left: 0,
                                            width: '100%',
                                            height: '100%',
                                            objectFit: 'contain',
                                            opacity: 0.6,
                                            mixBlendMode: 'multiply'
                                        }}
                                    />
                                )
                            ))}

                            {/* Canvas Overlay Placeholder */}
                            <div style={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                pointerEvents: 'none'
                            }}>
                                <Tag color="processing" style={{ fontSize: 11 }}>
                                    Interactive annotation canvas overlay
                                </Tag>
                            </div>
                        </div>
                    ) : (
                        <div style={{
                            width: '100%',
                            height: '100%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center'
                        }}>
                            <Space>
                                <Spin indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />} />
                                <Text type="secondary">Loading pseudo label...</Text>
                            </Space>
                        </div>
                    )}
                </div>

                {/* Pseudo Label Info */}
                {pseudoLabel && (
                    <div style={{ marginTop: 12 }}>
                        <Space size="middle" wrap>
                            <Tag color="blue">
                                Background: {pseudoLabel.background?.length || 0} × {pseudoLabel.background?.[0]?.length || 0}
                            </Tag>
                            <Tag color="green">
                                Layers: {pseudoLabel.layers?.length || 0}
                            </Tag>
                            <Tag color="orange">
                                Source: {pseudoLabel.image_path?.split('/').pop() || 'Unknown'}
                            </Tag>
                        </Space>
                    </div>
                )}
            </div>
        </Card>
    );
};