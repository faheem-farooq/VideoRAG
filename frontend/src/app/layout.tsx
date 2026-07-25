import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VideoRAG — multilingual video segment search",
  description:
    "Ask a question in any language, get back the exact timestamped segment of a video that answers it.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="min-h-full flex flex-col bg-neutral-950 text-neutral-100 font-sans">
        {children}
      </body>
    </html>
  );
}
