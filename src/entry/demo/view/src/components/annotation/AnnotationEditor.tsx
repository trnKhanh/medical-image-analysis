import type {AnnotationData, PseudoLabel} from "../../models";
import React, {useEffect, useRef, useState} from "react";
import {Button, Divider, Slider, Space, Spin, Tag, Tooltip, Typography} from "antd";
import {
    BgColorsOutlined,
    CheckOutlined,
    DeleteOutlined,
    LoadingOutlined,
    UndoOutlined,
    ZoomInOutlined,
    ZoomOutOutlined
} from "@ant-design/icons";

const { Text } = Typography;

export const AnnotationEditor: React.FC<{
    pseudoLabel: PseudoLabel;
    selectedImageContent: string;
    brushColor: string;
    isSubmitting: boolean;
    imagePath: string;
    onBrushColorChange: (color: string) => void;
    onSubmitAnnotation: (annotationData: AnnotationData) => void;
    onClose?: () => void;
}> = ({ pseudoLabel, selectedImageContent, brushColor, isSubmitting, imagePath, onBrushColorChange, onSubmitAnnotation, onClose }) => {

    const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const backgroundCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const drawing = useRef(false);
    const [strokes, setStrokes] = useState<ImageData[]>([]);
    const brushedPixels = useRef<Map<string, string>>(new Map());
    const [zoom, setZoom] = useState(1);
    const [brushSize, setBrushSize] = useState(5);
    const lastPosition = useRef<{ x: number; y: number } | null>(null);
    const [canvasResolution, setCanvasResolution] = useState({ width: 640, height: 320 });

    const createCustomCursor = (isEraser: boolean, size: number) => {
        const canvas = document.createElement('canvas');

        const overlayCanvas = overlayCanvasRef.current;
        let displayScale = 1;
        if (overlayCanvas) {
            const rect = overlayCanvas.getBoundingClientRect();
            displayScale = rect.width / overlayCanvas.width;
        }

        const effectiveBrushSize = size / zoom;
        const displaySize = effectiveBrushSize * displayScale * zoom;
        const cursorSize = Math.min(Math.max(displaySize * 2, 16), 64);

        canvas.width = cursorSize;
        canvas.height = cursorSize;
        const ctx = canvas.getContext('2d');

        if (!ctx) return 'crosshair';

        const centerX = cursorSize / 2;
        const centerY = cursorSize / 2;
        const radius = Math.max(displaySize, 4);

        if (isEraser) {
            ctx.fillStyle = '#ffffff';
            ctx.strokeStyle = '#333333';
            ctx.lineWidth = 2;

            const eraserWidth = radius * 1.5;
            const eraserHeight = radius * 0.8;
            ctx.fillRect(centerX - eraserWidth/2, centerY - eraserHeight/2, eraserWidth, eraserHeight);
            ctx.strokeRect(centerX - eraserWidth/2, centerY - eraserHeight/2, eraserWidth, eraserHeight);

            ctx.fillStyle = '#cccccc';
            ctx.fillRect(centerX - eraserWidth/2, centerY - eraserHeight/4, eraserWidth, eraserHeight/2);
        } else {
            ctx.fillStyle = 'rgba(51, 51, 51, 0.8)';
            ctx.strokeStyle = '#333333';
            ctx.lineWidth = 1;

            const handleLength = radius * 1.2;
            const handleWidth = radius * 0.3;
            ctx.fillRect(centerX - handleWidth/2, centerY + radius/2, handleWidth, handleLength);

            ctx.beginPath();
            ctx.arc(centerX, centerY, radius/2, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        }

        const dataURL = canvas.toDataURL();
        return `url(${dataURL}) ${centerX} ${centerY}, crosshair`;
    };

    const updateCanvasCursor = () => {
        const overlayCanvas = overlayCanvasRef.current;
        if (!overlayCanvas) return;

        const currentBrush = brushOptions.find(b => b.color === brushColor || b.hexColor === brushColor);
        const isErasing = currentBrush?.hexColor === "#ffffff";
        const customCursor = createCustomCursor(isErasing, brushSize);

        overlayCanvas.style.cursor = customCursor;

        const handleMouseMove = (e: MouseEvent) => {
            const rect = overlayCanvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const isInBounds = mouseX >= 0 && mouseX < overlayCanvas.width &&
                mouseY >= 0 && mouseY < overlayCanvas.height;

            overlayCanvas.style.cursor = isInBounds ? customCursor : 'not-allowed';
        };

        overlayCanvas.addEventListener('mousemove', handleMouseMove);

        return () => {
            overlayCanvas.removeEventListener('mousemove', handleMouseMove);
        };
    };

    useEffect(() => {
        return updateCanvasCursor();
    }, [brushColor, brushSize, updateCanvasCursor, zoom]);

    useEffect(() => {
        if (selectedImageContent && backgroundCanvasRef.current) {
            const backgroundCanvas = backgroundCanvasRef.current;
            const backgroundCtx = backgroundCanvas.getContext("2d");
            if (!backgroundCtx) return;

            const img = new Image();
            img.onload = () => {
                console.log('DEBUG: Loading background image to canvas');
                console.log('DEBUG: Original image dimensions:', img.width, 'x', img.height);

                const containerHeight = 320;
                const containerWidth = containerHeight * (img.width / img.height);

                const displayWidth = Math.min(containerWidth, window.innerWidth * 0.8); // Max 80% of screen width
                const displayHeight = displayWidth * (img.height / img.width);

                const resolution = {
                    width: Math.round(displayWidth),
                    height: Math.round(displayHeight)
                };

                setCanvasResolution(resolution);

                backgroundCanvas.width = resolution.width;
                backgroundCanvas.height = resolution.height;

                backgroundCtx.drawImage(img, 0, 0, resolution.width, resolution.height);

                if (overlayCanvasRef.current) {
                    const overlayCanvas = overlayCanvasRef.current;
                    overlayCanvas.width = resolution.width;
                    overlayCanvas.height = resolution.height;

                    const overlayCtx = overlayCanvas.getContext("2d");
                    if (overlayCtx) {
                        overlayCtx.clearRect(0, 0, resolution.width, resolution.height);
                    }
                }

                updateCanvasCursor();
            };
            img.onerror = (error) => {
                console.error('DEBUG: Failed to load background image:', error);
            };
            img.src = `data:image/jpeg;base64,${selectedImageContent}`;
        }
    }, [selectedImageContent]);

    useEffect(() => {
        if (pseudoLabel.layers && overlayCanvasRef.current && canvasResolution.width > 0) {
            const overlayCanvas = overlayCanvasRef.current;
            const overlayCtx = overlayCanvas.getContext("2d");
            if (!overlayCtx) return;

            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

            pseudoLabel.layers.forEach((layerData) => {
                if (typeof layerData === 'string') {
                    const img = new Image();
                    img.onload = () => {
                        overlayCtx.drawImage(img, 0, 0, overlayCanvas.width, overlayCanvas.height);
                    };
                    img.src = `data:image/png;base64,${layerData}`;
                } else {
                    const layerArr = layerData as any[][];
                    const h = layerArr.length;
                    const w = layerArr[0]?.length || 0;

                    const imgData = overlayCtx.createImageData(overlayCanvas.width, overlayCanvas.height);

                    for (let y = 0; y < overlayCanvas.height; y++) {
                        for (let x = 0; x < overlayCanvas.width; x++) {
                            const srcX = Math.floor((x / overlayCanvas.width) * w);
                            const srcY = Math.floor((y / overlayCanvas.height) * h);

                            if (srcY < h && srcX < w) {
                                const pix = layerArr[srcY][srcX];
                                const idx = (y * overlayCanvas.width + x) * 4;

                                if (Array.isArray(pix)) {
                                    imgData.data[idx] = pix[0] ?? 0;
                                    imgData.data[idx + 1] = pix[1] ?? 0;
                                    imgData.data[idx + 2] = pix[2] ?? 0;
                                    imgData.data[idx + 3] = pix[3] ?? 0;
                                } else {
                                    imgData.data[idx] = pix;
                                    imgData.data[idx + 1] = pix;
                                    imgData.data[idx + 2] = pix;
                                    imgData.data[idx + 3] = pix > 0 ? 128 : 0;
                                }
                            }
                        }
                    }
                    overlayCtx.putImageData(imgData, 0, 0);
                }
            });

            console.log('DEBUG: Pseudo labels drawn at canvas resolution:', overlayCanvas.width, 'x', overlayCanvas.height);
        }
    }, [pseudoLabel.layers, canvasResolution]);

    const drawLine = (overlayCtx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) => {
        const radius = brushSize / zoom;
        const currentBrush = brushOptions.find(b => b.color === brushColor || b.hexColor === brushColor);
        const isErasing = currentBrush?.hexColor === "#ffffff";

        overlayCtx.lineCap = "round";
        overlayCtx.lineJoin = "round";
        overlayCtx.lineWidth = radius * 2;

        if (isErasing) {
            overlayCtx.globalCompositeOperation = "destination-out";
            overlayCtx.strokeStyle = "rgba(0,0,0,1)";
        } else {
            overlayCtx.globalCompositeOperation = "source-over";
            overlayCtx.strokeStyle = currentBrush?.hexColor || brushColor; // Use solid color for overlay
        }

        overlayCtx.beginPath();
        overlayCtx.moveTo(x1, y1);
        overlayCtx.lineTo(x2, y2);
        overlayCtx.stroke();

        const distance = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
        const steps = Math.max(1, Math.ceil(distance));

        for (let i = 0; i <= steps; i++) {
            const t = steps === 0 ? 0 : i / steps;
            const x = Math.round(x1 + (x2 - x1) * t);
            const y = Math.round(y1 + (y2 - y1) * t);
            const key = `${x}_${y}`;

            if (isErasing) {
                brushedPixels.current.delete(key);
            } else {
                brushedPixels.current.set(key, currentBrush?.hexColor || brushColor);
            }
        }

        overlayCtx.globalCompositeOperation = "source-over";
    };

    const drawPoint = (overlayCtx: CanvasRenderingContext2D, x: number, y: number) => {
        const radius = brushSize / zoom;
        const currentBrush = brushOptions.find(b => b.color === brushColor || b.hexColor === brushColor);
        const isErasing = currentBrush?.hexColor === "#ffffff";
        const roundedX = Math.round(x);
        const roundedY = Math.round(y);
        const key = `${roundedX}_${roundedY}`;

        if (isErasing) {
            overlayCtx.globalCompositeOperation = "destination-out";
            overlayCtx.fillStyle = "rgba(0,0,0,1)";
            overlayCtx.beginPath();
            overlayCtx.arc(x, y, radius, 0, 2 * Math.PI);
            overlayCtx.fill();
            brushedPixels.current.delete(key);
        } else {
            const existingColor = brushedPixels.current.get(key);
            if (existingColor && existingColor === (currentBrush?.hexColor || brushColor)) return;

            overlayCtx.globalCompositeOperation = "source-over";
            overlayCtx.fillStyle = currentBrush?.hexColor || brushColor; // Use solid color for overlay
            overlayCtx.beginPath();
            overlayCtx.arc(x, y, radius, 0, 2 * Math.PI);
            overlayCtx.fill();
            brushedPixels.current.set(key, currentBrush?.hexColor || brushColor);
        }

        overlayCtx.globalCompositeOperation = "source-over";
    };

    const canvasToBase64 = (canvas: HTMLCanvasElement): string => {
        try {

            if (!canvas || canvas.width === 0 || canvas.height === 0) {
                throw new Error('Invalid canvas dimensions');
            }

            const dataURL = canvas.toDataURL('image/png', 1.0);

            if (!dataURL || dataURL === 'data:,') {
                throw new Error('Failed to generate dataURL from canvas');
            }

            const base64 = dataURL.split(',')[1]; // Remove the "data:image/png;base64," prefix

            console.log('DEBUG: Canvas to base64 conversion:');
            console.log('  - Canvas size:', canvas.width, 'x', canvas.height);
            console.log('  - DataURL length:', dataURL.length);
            console.log('  - Base64 length:', base64.length);
            console.log('  - DataURL starts with:', dataURL.substring(0, 50));
            console.log('  - Base64 first 50 chars:', base64.substring(0, 50));

            if (!base64 || base64.length < 100) {
                throw new Error(`Generated base64 is too short (${base64 ? base64.length : 0} chars)`);
            }

            return base64;
        } catch (error) {
            console.error('DEBUG: Error in canvasToBase64:', error);
            console.error('DEBUG: Canvas state:', {
                canvas: !!canvas,
                width: canvas?.width,
                height: canvas?.height,
                context: !!canvas?.getContext('2d')
            });
            throw error;
        }
    };

    const getAnnotationDataForSubmit = () => {
        console.log('DEBUG: Starting getAnnotationDataForSubmit');

        const backgroundCanvas = backgroundCanvasRef.current;
        const overlayCanvas = overlayCanvasRef.current;

        if (!backgroundCanvas) {
            console.error('DEBUG: Background canvas not found');
            return null;
        }

        if (!overlayCanvas) {
            console.error('DEBUG: Overlay canvas not found');
            return null;
        }

        let backgroundBase64;
        try {
            backgroundBase64 = canvasToBase64(backgroundCanvas);
        } catch (error) {
            console.error('DEBUG: Failed to get background base64:', error);
            return null;
        }

        let overlayMaskBase64;
        try {
            overlayMaskBase64 = canvasToBase64(overlayCanvas);
        } catch (error) {
            console.error('DEBUG: Failed to get overlay mask base64:', error);
            return null;
        }

        const result = {
            image_path: imagePath,
            background: backgroundBase64,
            layers: [overlayMaskBase64]
        };

        if (!result.background || !result.layers[0]) {
            console.error('DEBUG: CRITICAL - Missing background or layer data!');
            return null;
        }

        return result;
    };

    const handleSubmitAnnotation = async () => {

        const annotationData = getAnnotationDataForSubmit();
        if (annotationData) {
            try {
                await onSubmitAnnotation(annotationData);
            } catch (error) {
                console.error('DEBUG: Submission failed:', error);
            }
        } else {
            console.error('DEBUG: Failed to prepare annotation data');
        }
    };


    useEffect(() => {
        const overlayCanvas = overlayCanvasRef.current;
        if (!overlayCanvas) return;

        const overlayCtx = overlayCanvas.getContext("2d");
        if (!overlayCtx) return;

        const getMousePos = (e: MouseEvent) => {
            const rect = overlayCanvas.getBoundingClientRect();

            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;

            const isInBounds = canvasX >= 0 && canvasX < overlayCanvas.width &&
                canvasY >= 0 && canvasY < overlayCanvas.height;

            const clampedX = Math.max(0, Math.min(canvasX, overlayCanvas.width - 1));
            const clampedY = Math.max(0, Math.min(canvasY, overlayCanvas.height - 1));

            return {
                x: clampedX,
                y: clampedY,
                isInBounds: isInBounds
            };
        };

        const handleMouseDown = (e: MouseEvent) => {
            const pos = getMousePos(e);

            if (!pos.isInBounds) {
                console.log('DEBUG: Mouse click outside background bounds, ignoring');
                return;
            }

            drawing.current = true;
            setStrokes((prev) => [...prev, overlayCtx.getImageData(0, 0, overlayCanvas.width, overlayCanvas.height)]);

            lastPosition.current = { x: pos.x, y: pos.y };
            drawPoint(overlayCtx, pos.x, pos.y);
        };

        const handleMouseMove = (e: MouseEvent) => {
            if (!drawing.current || !lastPosition.current) return;

            const currentPos = getMousePos(e);

            if (!currentPos.isInBounds) {
                console.log('DEBUG: Mouse moved outside background bounds, stopping draw');
                drawing.current = false;
                lastPosition.current = null;
                return;
            }

            drawLine(overlayCtx, lastPosition.current.x, lastPosition.current.y, currentPos.x, currentPos.y);
            lastPosition.current = { x: currentPos.x, y: currentPos.y };
        };

        const handleMouseUp = () => {
            drawing.current = false;
            lastPosition.current = null;
        };

        overlayCanvas.addEventListener("mousedown", handleMouseDown);
        overlayCanvas.addEventListener("mousemove", handleMouseMove);
        overlayCanvas.addEventListener("mouseup", handleMouseUp);
        overlayCanvas.addEventListener("mouseleave", handleMouseUp);

        return () => {
            overlayCanvas.removeEventListener("mousedown", handleMouseDown);
            overlayCanvas.removeEventListener("mousemove", handleMouseMove);
            overlayCanvas.removeEventListener("mouseup", handleMouseUp);
            overlayCanvas.removeEventListener("mouseleave", handleMouseUp);
        };
    }, [brushColor, brushSize, drawLine, drawPoint, zoom]);

    const handleUndo = () => {
        const overlayCanvas = overlayCanvasRef.current;
        const overlayCtx = overlayCanvas?.getContext("2d");
        if (!overlayCanvas || !overlayCtx || strokes.length === 0) return;

        const last = strokes[strokes.length - 1];
        overlayCtx.putImageData(last, 0, 0);
        setStrokes((prev) => prev.slice(0, -1));
    };

    const handleReset = () => {
        const overlayCanvas = overlayCanvasRef.current;
        const overlayCtx = overlayCanvas?.getContext("2d");

        if (overlayCanvas && overlayCtx) {
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

            // Restore original pseudo label layers after reset
            if (pseudoLabel.layers) {
                pseudoLabel.layers.forEach((layerData) => {
                    if (typeof layerData === 'string') {
                        const img = new Image();
                        img.onload = () => {
                            overlayCtx.drawImage(img, 0, 0, overlayCanvas.width, overlayCanvas.height);
                        };
                        img.src = `data:image/png;base64,${layerData}`;
                    } else {
                        const layerArr = layerData as any[][];
                        const h = layerArr.length;
                        const w = layerArr[0]?.length || 0;

                        const imgData = overlayCtx.createImageData(overlayCanvas.width, overlayCanvas.height);

                        for (let y = 0; y < overlayCanvas.height; y++) {
                            for (let x = 0; x < overlayCanvas.width; x++) {
                                const srcX = Math.floor((x / overlayCanvas.width) * w);
                                const srcY = Math.floor((y / overlayCanvas.height) * h);

                                if (srcY < h && srcX < w) {
                                    const pix = layerArr[srcY][srcX];
                                    const idx = (y * overlayCanvas.width + x) * 4;

                                    if (Array.isArray(pix)) {
                                        imgData.data[idx] = pix[0] ?? 0;
                                        imgData.data[idx + 1] = pix[1] ?? 0;
                                        imgData.data[idx + 2] = pix[2] ?? 0;
                                        imgData.data[idx + 3] = pix[3] ?? 0;
                                    } else {
                                        imgData.data[idx] = pix;
                                        imgData.data[idx + 1] = pix;
                                        imgData.data[idx + 2] = pix;
                                        imgData.data[idx + 3] = pix > 0 ? 128 : 0;
                                    }
                                }
                            }
                        }
                        overlayCtx.putImageData(imgData, 0, 0);
                    }
                });
            }
        }

        setStrokes([]);
        brushedPixels.current.clear();
    };

    const brushOptions = [
        {
            color: "rgba(255, 0, 0, 0.03)",
            hexColor: "#ff0000",
            name: "Class 1 (Red)",
            bgColor: "#ff4d4f"
        },
        {
            color: "rgba(1, 255, 0, 0.03)",
            hexColor: "#00ff00",
            name: "Class 2 (Green)",
            bgColor: "#52c41a"
        },
        {
            color: "rgba(255, 255, 255, 1.0)",
            hexColor: "#ffffff",
            name: "Eraser",
            bgColor: "#fff",
            textColor: "#000"
        },
    ];

    return (
        <>
            <div style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16, flexWrap: "wrap" }}>
                    <Space size="middle" wrap>
                        <Space>
                            <BgColorsOutlined style={{ color: "#666" }} />
                            <Text type="secondary" style={{ fontSize: 12 }}>Annotation Classes:</Text>
                            <Space size="small">
                                {brushOptions.map(b => (
                                    <Tooltip key={b.color} title={b.name}>
                                        <Button
                                            size="small" shape="circle"
                                            onClick={() => onBrushColorChange(b.color)}
                                            style={{
                                                backgroundColor: b.bgColor,
                                                borderColor: (brushColor === b.color || brushColor === b.hexColor) ? "#333" : "#d9d9d9",
                                                borderWidth: (brushColor === b.color || brushColor === b.hexColor) ? 2 : 1,
                                                color: b.textColor || "#fff",
                                                width: 24, height: 24, minWidth: 24,
                                            }}
                                        />
                                    </Tooltip>
                                ))}
                            </Space>
                        </Space>

                        <Space>
                            <Tooltip title="Zoom In">
                                <Button icon={<ZoomInOutlined />} onClick={() => setZoom(prev => Math.min(prev + 0.25, 5))} />
                            </Tooltip>
                            <Tooltip title="Zoom Out">
                                <Button icon={<ZoomOutOutlined />} onClick={() => setZoom(prev => Math.max(prev - 0.25, 0.25))} />
                            </Tooltip>
                            <Text type="secondary">Zoom: {Math.round(zoom * 100)}%</Text>
                        </Space>

                        <Space>
                            <Text type="secondary">Brush Size:</Text>
                            <Slider
                                min={1} max={20} step={1}
                                value={brushSize}
                                onChange={setBrushSize}
                                style={{ width: 120 }}
                            />
                        </Space>

                        <Tooltip title="Undo">
                            <Button icon={<UndoOutlined />} onClick={handleUndo} />
                        </Tooltip>
                        <Tooltip title="Reset">
                            <Button icon={<DeleteOutlined />} onClick={handleReset} />
                        </Tooltip>

                        <Button type="primary" icon={isSubmitting ? <LoadingOutlined /> : <CheckOutlined />}
                                loading={isSubmitting} onClick={handleSubmitAnnotation}>
                            {isSubmitting ? "Submitting..." : "Accept"}
                        </Button>
                    </Space>
                </div>
                <Divider style={{ margin: 0 }} />
            </div>

            <div style={{ padding: 16, backgroundColor: "#fafafa", borderRadius: 6, border: "1px solid #f0f0f0" }}>
                <div style={{
                    position: "relative",
                    width: `${canvasResolution.width}px`,
                    height: `${canvasResolution.height}px`,
                    backgroundColor: "#f5f5f5",
                    borderRadius: 6,
                    overflow: "hidden",
                    border: "1px solid #e8e8e8",
                    transform: `scale(${zoom})`,
                    transformOrigin: "center center",
                    margin: "0 auto"
                }}>
                    {pseudoLabel.background ? (
                        <>
                            {/* Background image from pseudoLabel */}
                            <img
                                src={`data:image/png;base64,${pseudoLabel.background}`}
                                alt="Background"
                                style={{
                                    position: "absolute",
                                    top: 0,
                                    left: 0,
                                    width: "100%",
                                    height: "100%",
                                    objectFit: "fill"
                                }}
                            />

                            {/* Overlay canvas - now the main interactive brush canvas */}
                            <canvas
                                ref={overlayCanvasRef}
                                width={canvasResolution.width}
                                height={canvasResolution.height}
                                style={{
                                    position: "absolute",
                                    top: 0,
                                    left: 0,
                                    width: "100%",
                                    height: "100%",
                                    opacity: 0.6,
                                    mixBlendMode: "multiply"
                                }}
                            />

                            {/* Hidden canvas for background processing */}
                            <canvas
                                ref={backgroundCanvasRef}
                                width={canvasResolution.width}
                                height={canvasResolution.height}
                                style={{ display: "none" }}
                            />
                        </>
                    ) : (
                        <div style={{
                            width: "100%", height: "100%",
                            display: "flex", alignItems: "center", justifyContent: "center"
                        }}>
                            <Space>
                                <Spin indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />} />
                                <Text type="secondary">Loading pseudo labels...</Text>
                            </Space>
                        </div>
                    )}
                </div>

                {pseudoLabel.background && (
                    <div style={{ marginTop: 12 }}>
                        <Space size="middle" wrap>
                            <Tag color="blue">Background: Loaded</Tag>
                            <Tag color="green">Pseudo Labels: {pseudoLabel.layers?.length || 0}</Tag>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                                <Text type="secondary" style={{ fontSize: 11 }}>Classes:</Text>
                                <Tag color="#ff4d4f">🔴 Class 1 (Red)</Tag>
                                <Tag color="#52c41a">🟢 Class 2 (Green)</Tag>
                                <Tag>⚪ Background</Tag>
                            </div>
                        </Space>
                    </div>
                )}
            </div>
        </>
    );
};