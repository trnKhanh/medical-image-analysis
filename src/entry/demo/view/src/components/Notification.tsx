import React from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';

interface NotificationProps {
    type: 'error' | 'success';
    message: string;
    onClose?: () => void;
}

export const Notification: React.FC<NotificationProps> = ({ type, message, onClose }) => {
    const isError = type === 'error';

    return (
        <div className="max-w-7xl mx-auto px-4 py-2">
            <div className={`flex items-center space-x-2 p-4 border rounded-md ${
                isError
                    ? 'bg-red-100 border-red-300'
                    : 'bg-green-100 border-green-300'
            }`}>
                {isError ? (
                    <AlertCircle className="h-5 w-5 text-red-600" />
                ) : (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                )}
                <span className={isError ? 'text-red-800' : 'text-green-800'}>
                    {message}
                </span>
                {onClose && (
                    <button
                        onClick={onClose}
                        className="ml-auto text-gray-500 hover:text-gray-700"
                    >
                        ×
                    </button>
                )}
            </div>
        </div>
    );
};
