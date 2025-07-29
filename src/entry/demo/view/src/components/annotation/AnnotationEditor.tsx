import type { PseudoLabel } from "../../models";
import React, { useRef, useEffect, useState } from "react";
import { Button, Space, Typography, Tooltip, Tag, Spin, Divider, Slider } from "antd";
import {
    CheckOutlined, BgColorsOutlined, LoadingOutlined,
    UndoOutlined, DeleteOutlined, ZoomInOutlined, ZoomOutOutlined
} from "@ant-design/icons";

const { Text } = Typography;

export const AnnotationEditor: React.FC<{
    pseudoLabel: PseudoLabel;
    selectedImageContent: string;
    brushColor: string;
    isSubmitting: boolean;
    imageIndex: number;
    onBrushColorChange: (color: string) => void;
    onSubmitAnnotation: (imageIndex: number, annotationData: any) => void;
}> = ({ pseudoLabel, selectedImageContent, brushColor, isSubmitting, imageIndex, onBrushColorChange, onSubmitAnnotation }) => {
    const { layers, background } = pseudoLabel;

    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const drawing = useRef(false);
    const [strokes, setStrokes] = useState<ImageData[]>([]);
    const brushedPixels = useRef<Map<string, string>>(new Map());
    const [zoom, setZoom] = useState(1);
    const [brushSize, setBrushSize] = useState(5);
    const lastPosition = useRef<{ x: number; y: number } | null>(null);

    const drawLine = (ctx: CanvasRenderingContext2D, x1: number, y1: number, x2: number, y2: number) => {
        const radius = brushSize * zoom;
        const isErasing = brushColor.toLowerCase() === "#ffffff";

        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.lineWidth = radius * 2;

        if (isErasing) {
            ctx.globalCompositeOperation = "destination-out";
            ctx.strokeStyle = "rgba(0,0,0,1)";
        } else {
            ctx.globalCompositeOperation = "source-over";
            ctx.globalAlpha = 0.8;
            ctx.strokeStyle = brushColor;
        }

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();

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
                brushedPixels.current.set(key, brushColor);
            }
        }

        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1.0;
    };

    const drawPoint = (ctx: CanvasRenderingContext2D, x: number, y: number) => {
        const radius = brushSize * zoom;
        const isErasing = brushColor.toLowerCase() === "#ffffff";
        const roundedX = Math.round(x);
        const roundedY = Math.round(y);
        const key = `${roundedX}_${roundedY}`;

        if (isErasing) {
            ctx.globalCompositeOperation = "destination-out";
            ctx.fillStyle = "rgba(0,0,0,1)";
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
            brushedPixels.current.delete(key);
        } else {
            const existingColor = brushedPixels.current.get(key);
            if (existingColor && existingColor === brushColor) return;

            ctx.globalCompositeOperation = "source-over";
            ctx.globalAlpha = 0.8;
            ctx.fillStyle = brushColor;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();
            brushedPixels.current.set(key, brushColor);
        }

        ctx.globalCompositeOperation = "source-over";
        ctx.globalAlpha = 1.0;
    };

    const getAnnotationData = () => {
        const canvas = canvasRef.current;
        if (!canvas) return null;

        const ctx = canvas.getContext("2d");
        if (!ctx) return null;

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        const annotationMask = [];
        for (let y = 0; y < canvas.height; y++) {
            const row = [];
            for (let x = 0; x < canvas.width; x++) {
                const index = (y * canvas.width + x) * 4;
                const alpha = imageData.data[index + 3];
                row.push(alpha > 0 ? 1 : 0);
            }
            annotationMask.push(row);
        }

        return {
            image_index: imageIndex,
            annotation_mask: annotationMask,
            brush_strokes: Array.from(brushedPixels.current.entries()).map(([position, color]) => ({
                position,
                color,
                brush_size: brushSize
            }))
        };
    };

    const handleSubmitAnnotation = () => {
        const annotationData = getAnnotationData();
        if (annotationData) {
            onSubmitAnnotation(imageIndex, annotationData);
        }
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const getMousePos = (e: MouseEvent) => {
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            return {
                x: (e.clientX - rect.left) * scaleX,
                y: (e.clientY - rect.top) * scaleY
            };
        };

        const handleMouseDown = (e: MouseEvent) => {
            drawing.current = true;
            setStrokes((prev) => [...prev, ctx.getImageData(0, 0, canvas.width, canvas.height)]);

            const pos = getMousePos(e);
            lastPosition.current = pos;
            drawPoint(ctx, pos.x, pos.y);
        };

        const handleMouseMove = (e: MouseEvent) => {
            if (!drawing.current || !lastPosition.current) return;

            const currentPos = getMousePos(e);
            drawLine(ctx, lastPosition.current.x, lastPosition.current.y, currentPos.x, currentPos.y);
            lastPosition.current = currentPos;
        };

        const handleMouseUp = () => {
            drawing.current = false;
            lastPosition.current = null;
        };

        canvas.addEventListener("mousedown", handleMouseDown);
        canvas.addEventListener("mousemove", handleMouseMove);
        canvas.addEventListener("mouseup", handleMouseUp);
        canvas.addEventListener("mouseleave", handleMouseUp);

        return () => {
            canvas.removeEventListener("mousedown", handleMouseDown);
            canvas.removeEventListener("mousemove", handleMouseMove);
            canvas.removeEventListener("mouseup", handleMouseUp);
            canvas.removeEventListener("mouseleave", handleMouseUp);
        };
    }, [brushColor, brushSize, zoom]);

    const handleUndo = () => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx || strokes.length === 0) return;

        const last = strokes[strokes.length - 1];
        ctx.putImageData(last, 0, 0);
        setStrokes((prev) => prev.slice(0, -1));

        // Note: This simple undo doesn't restore the exact brushedPixels state
        // For more accurate undo, you'd need to store the brushedPixels state as well
    };

    const handleReset = () => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setStrokes([]);
        brushedPixels.current.clear();
    };

    const backgroundImageSrc = selectedImageContent;

    const layerOverlays = React.useMemo(() => {
        return layers.map((layerArr) => {
            const h = layerArr.length;
            const w = layerArr[0]?.length || 0;
            const canvas = document.createElement("canvas");
            canvas.width = w;
            canvas.height = h;
            const ctx = canvas.getContext("2d");
            if (!ctx) return null;
            const imgData = ctx.createImageData(w, h);
            for (let y = 0; y < h; ++y) {
                for (let x = 0; x < w; ++x) {
                    const pix = layerArr[y][x];
                    const idx = (y * w + x) * 4;
                    if (Array.isArray(pix)) {
                        imgData.data[idx]     = pix[0] ?? 0;
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
            ctx.putImageData(imgData, 0, 0);
            return canvas.toDataURL();
        }).filter(Boolean) as string[];
    }, [layers]);

    const brushOptions = [
        { color: "#ff0000", name: "Red", bgColor: "#ff4d4f" },
        { color: "#00ff00", name: "Green", bgColor: "#52c41a" },
        { color: "#ffffff", name: "Eraser", bgColor: "#fff", textColor: "#000" },
    ];

    return (
        <>
            <div style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16, flexWrap: "wrap" }}>
                    <Space size="middle" wrap>
                        <Space>
                            <BgColorsOutlined style={{ color: "#666" }} />
                            <Text type="secondary" style={{ fontSize: 12 }}>Brush:</Text>
                            <Space size="small">
                                {brushOptions.map(b => (
                                    <Tooltip key={b.color} title={b.name}>
                                        <Button
                                            size="small" shape="circle"
                                            onClick={() => onBrushColorChange(b.color)}
                                            style={{
                                                backgroundColor: b.bgColor,
                                                borderColor: brushColor === b.color ? "#333" : "#d9d9d9",
                                                borderWidth: brushColor === b.color ? 2 : 1,
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
                    width: "100%", height: 320,
                    backgroundColor: "#f5f5f5", borderRadius: 6,
                    overflow: "hidden", border: "1px solid #e8e8e8"
                }}>
                    {backgroundImageSrc ? (
                        <>
                            <img
                                src={`data:image/jpeg;base64,${backgroundImageSrc}`}
                                alt="Background"
                                style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", objectFit: "contain" }}
                            />

                            {layerOverlays.map((src, i) => (
                                <img key={i} src={src} alt={`Layer ${i}`}
                                     style={{
                                         position: "absolute", top: 0, left: 0,
                                         width: "100%", height: "100%",
                                         objectFit: "contain",
                                         opacity: 0.6,
                                         mixBlendMode: "multiply"
                                     }}
                                />
                            ))}

                            <canvas
                                ref={canvasRef}
                                width={640}
                                height={320}
                                style={{
                                    position: "absolute", top: 0, left: 0,
                                    width: "100%", height: "100%",
                                    cursor: "crosshair"
                                }}
                            />
                        </>
                    ) : (
                        <div style={{
                            width: "100%", height: "100%",
                            display: "flex", alignItems: "center", justifyContent: "center"
                        }}>
                            <Space>
                                <Spin indicator={<LoadingOutlined style={{ fontSize: 24 }} spin />} />
                                <Text type="secondary">Loading image...</Text>
                            </Space>
                        </div>
                    )}
                </div>

                {background && (
                    <div style={{ marginTop: 12 }}>
                        <Space size="middle" wrap>
                            <Tag color="blue">Background: {background.length} × {background[0]?.length}</Tag>
                            <Tag color="green">Layers: {layers.length}</Tag>
                        </Space>
                    </div>
                )}
            </div>
        </>
    );
};