import {Card, Upload, Button, Space, Typography, Divider, Row, Col, ConfigProvider} from 'antd';
import { UploadOutlined, InboxOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import React, { useState } from 'react';

const { Title, Text } = Typography;

interface FileUploadPanelProps {
    trainFiles: FileList | null;
    poolFiles: FileList | null;
    loading: { train: boolean; pool: boolean };
    onTrainFilesChange: (files: FileList | null) => void;
    onPoolFilesChange: (files: FileList | null) => void;
    onUploadTrain: () => void;
    onUploadPool: () => void;
}

export const FileUploadPanel: React.FC<FileUploadPanelProps> = ({
                                                                    trainFiles,
                                                                    poolFiles,
                                                                    loading,
                                                                    onTrainFilesChange,
                                                                    onPoolFilesChange,
                                                                    onUploadTrain,
                                                                    onUploadPool
                                                                }) => {
    const [trainFileList, setTrainFileList] = useState<any[]>([]);
    const [poolFileList, setPoolFileList] = useState<any[]>([]);

    const trainUploadProps: UploadProps = {
        name: 'trainFiles',
        multiple: true,
        accept: 'image/*',
        beforeUpload: () => false,
        fileList: trainFileList,
        onChange: (info) => {
            setTrainFileList(info.fileList);
            const files = info.fileList.map(file => file.originFileObj).filter(Boolean);
            const fileList = new DataTransfer();
            files.forEach(file => file && fileList.items.add(file));
            onTrainFilesChange(fileList.files);
        },
        showUploadList: {
            showPreviewIcon: true,
            showRemoveIcon: true,
        },
        listType: 'picture',
    };

    const poolUploadProps: UploadProps = {
        name: 'poolFiles',
        multiple: true,
        accept: 'image/*',
        beforeUpload: () => false,
        fileList: poolFileList,
        onChange: (info) => {
            setPoolFileList(info.fileList);
            const files = info.fileList.map(file => file.originFileObj).filter(Boolean);
            const fileList = new DataTransfer();
            files.forEach(file => file && fileList.items.add(file));
            onPoolFilesChange(fileList.files);
        },
        showUploadList: {
            showPreviewIcon: true,
            showRemoveIcon: true,
        },
        listType: 'picture',
    };

    const handleUploadTrain = () => {
        onUploadTrain();
        setTrainFileList([]);
        onTrainFilesChange(null);
    };

    const handleUploadPool = () => {
        onUploadPool();
        setPoolFileList([]);
        onPoolFilesChange(null);
    };

    return (
        <Card
            size="default"
            style={{
                borderRadius: 8,
                boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
            }}
        >
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
                {/* Header */}
                <Space align="center">
                    <UploadOutlined style={{ fontSize: 20, color: '#666' }} />
                    <Title level={4} style={{ margin: 0 }}>
                        Upload Images
                    </Title>
                </Space>

                <Divider style={{ margin: 0 }} />

                <Row gutter={16}>
                    <Col xs={24} md={12}>
                        <div>
                            <Text strong style={{ display: 'block', marginBottom: 12 }}>
                                Training Images
                            </Text>
                            <Upload.Dragger {...trainUploadProps} style={{ height: '120px' }}>
                                <p className="ant-upload-drag-icon">
                                    <InboxOutlined style={{ fontSize: 32, color: '#1890ff' }} />
                                </p>
                                <p className="ant-upload-text" style={{ fontSize: 12 }}>
                                    Click or drag training images here
                                </p>
                                <p className="ant-upload-hint" style={{ fontSize: 11 }}>
                                    Multiple image files supported
                                </p>
                            </Upload.Dragger>

                            <Button className="!bg-blue-300 !border-blue-300 mt-3"
                                    type="primary"
                                    icon={<UploadOutlined />}
                                    loading={loading.train}
                                    disabled={!trainFiles || trainFiles.length === 0}
                                    onClick={handleUploadTrain}
                                    block
                                    size="middle"
                                    style={{ marginTop: 12 }}
                            >
                                {loading.train ? 'Uploading...' : 'Upload Training'}
                            </Button>
                        </div>
                    </Col>

                    {/* Pool Images Section */}
                    <Col xs={24} md={12}>
                        <div>
                            <Text strong style={{ display: 'block', marginBottom: 12 }}>
                                Pool Images
                            </Text>
                            <Upload.Dragger {...poolUploadProps} style={{ height: '120px' }}>
                                <p className="ant-upload-drag-icon">
                                    <InboxOutlined style={{ fontSize: 32, color: '#52c41a' }} />
                                </p>
                                <p className="ant-upload-text" style={{ fontSize: 12 }}>
                                    Click or drag pool images here
                                </p>
                                <p className="ant-upload-hint" style={{ fontSize: 11 }}>
                                    Used for the image pool
                                </p>
                            </Upload.Dragger>

                            <ConfigProvider
                                theme={{
                                    token: {
                                        colorPrimary: '#52c41a',
                                    },
                                }}
                            >
                                <Button
                                    type="primary"
                                    icon={<UploadOutlined />}
                                    loading={loading.pool}
                                    disabled={!poolFiles || poolFiles.length === 0}
                                    onClick={handleUploadPool}
                                    block
                                    size="middle"
                                    style={{ marginTop: 12 }}
                                >
                                    {loading.pool ? 'Uploading...' : 'Upload Pool'}
                                </Button>
                            </ConfigProvider>
                        </div>
                    </Col>
                </Row>
            </Space>
        </Card>
    );
};