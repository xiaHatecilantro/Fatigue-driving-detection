import { useState } from "react";

interface UploadPanelProps {
  title: string;
  accept: string;
  busy?: boolean;
  onSubmit: (file: File) => Promise<void>;
}

export default function UploadPanel({
  title,
  accept,
  busy = false,
  onSubmit,
}: UploadPanelProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  async function handleSubmit() {
    if (!selectedFile) {
      return;
    }
    await onSubmit(selectedFile);
  }

  return (
    <section className="panel">
      <div className="panel-header">
        <h3>{title}</h3>
      </div>
      <div className="upload-panel">
        <input
          type="file"
          accept={accept}
          onChange={(event) =>
            setSelectedFile(event.target.files?.[0] ?? null)
          }
        />
        <button className="action-button" disabled={!selectedFile || busy} onClick={handleSubmit}>
          {busy ? "处理中..." : "开始推理"}
        </button>
        <div className="file-caption">
          {selectedFile ? selectedFile.name : "尚未选择文件"}
        </div>
      </div>
    </section>
  );
}
