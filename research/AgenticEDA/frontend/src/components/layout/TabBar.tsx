"use client";

interface Props {
  activeTab: "notebook" | "story";
  onTabChange: (tab: "notebook" | "story") => void;
  dirty?: boolean;
  onConfirm?: () => void;
  onHistory?: () => void;
}

export default function TabBar({ activeTab, onTabChange, dirty, onConfirm, onHistory }: Props) {
  const tabClass = (tab: "notebook" | "story") =>
    `px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
      activeTab === tab
        ? "border-blue-600 text-blue-600"
        : "border-transparent text-gray-500 hover:text-gray-700"
    }`;

  return (
    <div className="flex items-center border-b border-gray-200 px-4 bg-white">
      <button className={tabClass("notebook")} onClick={() => onTabChange("notebook")}>
        Notebook
      </button>
      <button className={tabClass("story")} onClick={() => onTabChange("story")}>
        Story
      </button>

      <div className="ml-auto flex items-center gap-2">
        <button
          onClick={onHistory}
          className="p-2 text-gray-400 hover:text-gray-600 text-sm"
          title="Version history"
          aria-label="Version history"
        >
          &#9201;
        </button>
        {dirty && (
          <button
            onClick={onConfirm}
            className="px-3 py-1 text-xs rounded bg-green-600 text-white hover:bg-green-700 transition-colors"
          >
            Confirm Changes
          </button>
        )}
      </div>
    </div>
  );
}
