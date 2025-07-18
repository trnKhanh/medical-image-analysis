import type { PseudoLabel } from "../../models";
import React, { useRef, useEffect, useState } from "react";
import { Button, Space, Typography, Tooltip, Tag, Spin, Divider } from "antd";
import { CheckOutlined, BgColorsOutlined, LoadingOutlined, UndoOutlined, DeleteOutlined } from "@ant-design/icons";

const { Text } = Typography;

export const AnnotationEditor: React.FC<{
    pseudoLabel: PseudoLabel;
    selectedImageContent: string;
    brushColor: string;
    isSubmitting: boolean;
    onBrushColorChange: (color: string) => void;
    onSubmitAnnotation: () => void;
}> = ({ pseudoLabel, selectedImageContent, brushColor, isSubmitting, onBrushColorChange, onSubmitAnnotation }) => {
    const { layers, background } = pseudoLabel;

    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const drawing = useRef(false);
    const [strokes, setStrokes] = useState<ImageData[]>([]);

    // const draw = (ctx: CanvasRenderingContext2D, x: number, y: number) => {
    //     ctx.fillStyle = brushColor;
    //     ctx.beginPath();
    //     ctx.arc(x, y, 5, 0, 2 * Math.PI);
    //     ctx.fill();
    // };
    const draw = (ctx: CanvasRenderingContext2D, x: number, y: number) => {
        if (brushColor.toLowerCase() === "#ffffff") {
            // Eraser: clear a small area around the point
            ctx.clearRect(x - 5, y - 5, 10, 10);
        } else {
            ctx.fillStyle = brushColor;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fill();
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
    }, [brushColor, draw]);


    const handleUndo = () => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx || strokes.length === 0) return;

        const last = strokes[strokes.length - 1];
        ctx.putImageData(last, 0, 0);
        setStrokes((prev) => prev.slice(0, -1));
    };

    const handleReset = () => {
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d");
        if (!canvas || !ctx) return;

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        setStrokes([]);
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
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 16 }}>
                <Space size="middle">
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
                    <Tooltip title="Undo">
                        <Button icon={<UndoOutlined />} onClick={handleUndo} />
                    </Tooltip>
                    <Tooltip title="Reset">
                        <Button icon={<DeleteOutlined />} onClick={handleReset} />
                    </Tooltip>
                    <Button type="primary" icon={isSubmitting ? <LoadingOutlined /> : <CheckOutlined />}
                            loading={isSubmitting} onClick={onSubmitAnnotation}>
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
