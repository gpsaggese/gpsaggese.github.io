import DropZone from "@/components/upload/DropZone";
import AboutLauncher from "@/components/about/AboutLauncher";

export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-surface">
      <AboutLauncher />
      <h1 className="text-4xl font-extrabold tracking-tight text-on-surface mb-2 font-headline">
        AgenticEDA
      </h1>
      <p className="text-on-surface-variant mb-10 font-body">
        Upload a dataset to begin
      </p>
      <DropZone />
    </div>
  );
}
