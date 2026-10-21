import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { renderRoute } from "../test/utils";

describe("keyboard accessibility", () => {
  it("collapses the navigation rail with the keyboard", async () => {
    const user = userEvent.setup();
    renderRoute("/");
    const toggle = screen.getByRole("button", { name: "Collapse navigation" });
    expect(toggle).toHaveAttribute("aria-expanded", "true");
    toggle.focus();
    await user.keyboard("{Enter}");
    expect(screen.getByRole("button", { name: "Expand navigation" })).toHaveAttribute("aria-expanded", "false");
  });

  it("marks the current page in the navigation", () => {
    renderRoute("/history");
    const links = screen.getAllByRole("link", { name: /History/ });
    expect(links.some((l) => l.getAttribute("aria-current") === "page")).toBe(true);
  });

  it("supports arrow-key navigation between input tabs", async () => {
    const user = userEvent.setup();
    renderRoute("/predictions");
    const csvTab = screen.getByRole("tab", { name: /CSV upload/ });
    csvTab.focus();
    await user.keyboard("{ArrowRight}");
    const jsonTab = screen.getByRole("tab", { name: /Paste JSON/ });
    expect(jsonTab).toHaveAttribute("aria-selected", "true");
    expect(jsonTab).toHaveFocus();
  });

  it("opens and closes the mobile navigation with Escape", async () => {
    const user = userEvent.setup();
    renderRoute("/");
    await user.click(screen.getByRole("button", { name: "Open navigation" }));
    expect(screen.getByRole("dialog", { name: "Navigation" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Close navigation" })).toHaveFocus();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Navigation" })).not.toBeInTheDocument();
  });

  it("offers a skip link to the main content", () => {
    renderRoute("/");
    expect(screen.getByRole("link", { name: "Skip to content" })).toHaveAttribute("href", "#main");
  });
});
