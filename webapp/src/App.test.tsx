import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";

import App from "./App";

describe("App", () => {
  it("renders the upload hero text", () => {
    window.history.pushState({}, "", "/");
    render(<App />);
    expect(screen.getByText("上传一段网球视频，自动跑完整条动作分析链路")).toBeInTheDocument();
  });
});
