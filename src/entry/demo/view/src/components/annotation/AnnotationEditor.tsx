
import type {AnnotationData, PseudoLabel} from "../../models";
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
    onSubmitAnnotation: (annotationData: any) => void;
}> = ({ pseudoLabel, selectedImageContent, brushColor, isSubmitting, imageIndex, onBrushColorChange, onSubmitAnnotation }) => {
    const { layers, background } = pseudoLabel;

    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const drawing = useRef(false);
    const [strokes, setStrokes] = useState<ImageData[]>([]);
    const brushedPixels = useRef<Map<string, string>>(new Map());
    const [zoom, setZoom] = useState(1);
    const [brushSize, setBrushSize] = useState(5);

    const draw = (ctx: CanvasRenderingContext2D, x: number, y: number) => {
        const radius = brushSize * zoom;
        const roundedX = Math.round(x);
        const roundedY = Math.round(y);
        const key = `${roundedX}_${roundedY}`;
        const existingColor = brushedPixels.current.get(key);

        const isErasing = brushColor.toLowerCase() === "#ffffff";

        if (isErasing) {
            ctx.clearRect(x - radius, y - radius, radius * 2, radius * 2);
            brushedPixels.current.delete(key);
        } else {
            if (existingColor && existingColor === brushColor) return;

            ctx.clearRect(x - radius, y - radius, radius * 2, radius * 2);

            const prevAlpha = ctx.globalAlpha;
            ctx.globalAlpha = 0.5;

            ctx.fillStyle = brushColor;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, 2 * Math.PI);
            ctx.fill();

            ctx.globalAlpha = prevAlpha;

            brushedPixels.current.set(key, brushColor);
        }
    };

    const getAnnotationData = () => {
        const canvas = canvasRef.current;
        if (!canvas) return null;

        const ctx = canvas.getContext("2d");
        if (!ctx) return null;

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        const annotationMask: number[][] = [];
        const colorToClass: Record<string, number> = {
            "#ff0000": 1,
            "#00ff00": 2,
        };

        for (let y = 0; y < canvas.height; y++) {
            const row: number[] = [];
            for (let x = 0; x < canvas.width; x++) {
                const idx = (y * canvas.width + x) * 4;
                const r = imageData.data[idx];
                const g = imageData.data[idx + 1];
                const b = imageData.data[idx + 2];
                const a = imageData.data[idx + 3];

                let label = 0;
                if (a > 0) {
                    const hex = rgbToHex(r, g, b);
                    label = colorToClass[hex.toLowerCase()] ?? 0;
                }

                row.push(label);
            }
            annotationMask.push(row);
        }

        const outputCanvas = document.createElement("canvas");
        outputCanvas.width = canvas.width;
        outputCanvas.height = canvas.height;
        const outCtx = outputCanvas.getContext("2d");
        const outputData = outCtx!.createImageData(canvas.width, canvas.height);

        const classToColor: Record<number, [number, number, number]> = {
            1: [255, 0, 0],
            2: [0, 255, 0],
        };

        for (let y = 0; y < canvas.height; y++) {
            for (let x = 0; x < canvas.width; x++) {
                const label = annotationMask[y][x];
                const [r, g, b] = classToColor[label] ?? [0, 0, 0];
                const idx = (y * canvas.width + x) * 4;
                outputData.data[idx] = r;
                outputData.data[idx + 1] = g;
                outputData.data[idx + 2] = b;
                outputData.data[idx + 3] = label > 0 ? 255 : 0;
            }
        }

        outCtx!.putImageData(outputData, 0, 0);
        const layerBase64 = outputCanvas.toDataURL("image/png").split(",")[1];

        return {
            image_index: imageIndex,
            annotation_mask: annotationMask,
            brush_strokes: Array.from(brushedPixels.current.entries()).map(([position, color]) => ({
                position,
                color,
                brush_size: brushSize
            })),
            layer_base64: layerBase64
        };
    };

    const handleSubmitAnnotation = () => {
        const annotationData = getAnnotationData();
        if (annotationData) {
            onSubmitAnnotation(annotationData);
        }
    };

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        const handleMouseDown = (e: MouseEvent) => {
            drawing.current = true;
            setStrokes((prev) => [...prev, ctx.getImageData(0, 0, canvas.width, canvas.height)]);
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;
            draw(ctx, x, y);
        };

        const handleMouseMove = (e: MouseEvent) => {
            if (!drawing.current) return;
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX;
            const y = (e.clientY - rect.top) * scaleY;
            draw(ctx, x, y);
        };

        const handleMouseUp = () => {
            drawing.current = false;
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
        brushedPixels.current.clear(); // optionally restore pixels
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

function rgbToHex(r: number, g: number, b: number): string {
    return (
        "#" +
        [r, g, b]
            .map((x) => {
                const hex = x.toString(16);
                return hex.length === 1 ? "0" + hex : hex;
            })
            .join("")
    );
}
