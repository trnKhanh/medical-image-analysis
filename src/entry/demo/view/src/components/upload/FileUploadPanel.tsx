import React from 'react';
import { Card, Upload, Button, Space, Typography, Divider, Row, Col } from 'antd';
import { UploadOutlined, InboxOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';

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
    const trainUploadProps: UploadProps = {
        name: 'trainFiles',
        multiple: true,
        accept: 'image/*',
        beforeUpload: () => false, // Prevent auto upload
        onChange: (info) => {
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
        beforeUpload: () => false, // Prevent auto upload
        onChange: (info) => {
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

                {/* Horizontal Layout for Training and Pool Images */}
                <Row gutter={16}>
                    {/* Training Images Section */}
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
                                onClick={onUploadTrain}
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

                            <Button
                                className="!bg-green-300 !border-green-300 mt-3"
                                type="default"
                                icon={<UploadOutlined />}
                                loading={loading.pool}
                                disabled={!poolFiles || poolFiles.length === 0}
                                onClick={onUploadPool}
                                block
                                size="middle"
                            >
                                {loading.pool ? 'Uploading...' : 'Upload Pool'}
                            </Button>
                        </div>
                    </Col>
                </Row>
            </Space>
        </Card>
    );
};