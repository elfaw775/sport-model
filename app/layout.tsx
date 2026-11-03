import "./globals.css";

export const metadata = {
  title: "马球节病变检测 · YOLO11",
  description: "浏览器端 ONNX 推理：上传 X 光片，检测马球节病变",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
