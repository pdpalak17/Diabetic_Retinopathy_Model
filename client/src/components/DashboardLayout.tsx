import { useAuth } from "@/_core/hooks/useAuth";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { Sidebar, SidebarContent, SidebarFooter, SidebarHeader, SidebarInset, SidebarMenu, SidebarMenuButton, SidebarMenuItem, SidebarProvider, SidebarTrigger, useSidebar } from "@/components/ui/sidebar";
import { startLogin } from "@/const";
import { useIsMobile } from "@/hooks/useMobile";
import { FilePlus2, LayoutDashboard, LogOut, PanelLeft } from "lucide-react";
import { CSSProperties, useEffect, useRef, useState } from "react";
import { useLocation } from "wouter";
import { DashboardLayoutSkeleton } from "./DashboardLayoutSkeleton";

const menuItems = [
  { icon: LayoutDashboard, label: "Workspace", path: "/app" },
  { icon: FilePlus2, label: "New screening", path: "/app/new" },
];
const SIDEBAR_WIDTH_KEY = "retina-signal-sidebar-width";
const DEFAULT_WIDTH = 254;
const MIN_WIDTH = 210;
const MAX_WIDTH = 380;

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const [sidebarWidth, setSidebarWidth] = useState(() => Number(localStorage.getItem(SIDEBAR_WIDTH_KEY)) || DEFAULT_WIDTH);
  const { loading, user } = useAuth();

  useEffect(() => { localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth)); }, [sidebarWidth]);
  if (loading) return <DashboardLayoutSkeleton />;
  if (!user) {
    return (
      <div className="min-h-screen grid place-items-center p-5">
        <div className="blueprint-card w-full max-w-md rounded-2xl p-8 text-center">
          <p className="technical-label text-primary">secure workspace</p>
          <h1 className="mt-3 text-3xl font-bold tracking-tight">Sign in to review cases.</h1>
          <p className="mt-4 text-sm leading-6 text-muted-foreground">Your screening workspace uses Manus OAuth to protect user-owned case data and retinal images.</p>
          <Button onClick={() => startLogin()} className="button-press mt-7 w-full" size="lg">Continue with Manus</Button>
        </div>
      </div>
    );
  }

  return <SidebarProvider style={{ "--sidebar-width": `${sidebarWidth}px` } as CSSProperties}><DashboardLayoutContent setSidebarWidth={setSidebarWidth}>{children}</DashboardLayoutContent></SidebarProvider>;
}

function DashboardLayoutContent({ children, setSidebarWidth }: { children: React.ReactNode; setSidebarWidth: (width: number) => void }) {
  const { user, logout } = useAuth();
  const [location, setLocation] = useLocation();
  const { state, toggleSidebar } = useSidebar();
  const isCollapsed = state === "collapsed";
  const isMobile = useIsMobile();
  const [isResizing, setIsResizing] = useState(false);
  const sidebarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const move = (event: MouseEvent) => {
      if (!isResizing) return;
      const left = sidebarRef.current?.getBoundingClientRect().left ?? 0;
      const width = event.clientX - left;
      if (width >= MIN_WIDTH && width <= MAX_WIDTH) setSidebarWidth(width);
    };
    const up = () => setIsResizing(false);
    if (isResizing) { document.addEventListener("mousemove", move); document.addEventListener("mouseup", up); }
    return () => { document.removeEventListener("mousemove", move); document.removeEventListener("mouseup", up); };
  }, [isResizing, setSidebarWidth]);

  const active = menuItems.find(item => item.path === location)?.label ?? (location.startsWith("/app/cases/") ? "Case analysis" : "Workspace");
  return (
    <>
      <div ref={sidebarRef} className="relative no-print">
        <Sidebar collapsible="icon" className="border-r-0 bg-sidebar/95 backdrop-blur" disableTransition={isResizing}>
          <SidebarHeader className="h-20 justify-center px-3">
            <div className="flex items-center gap-3">
              <button onClick={toggleSidebar} className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-border bg-background/80 transition-colors hover:bg-secondary" aria-label="Toggle navigation"><PanelLeft className="h-4 w-4" /></button>
              {!isCollapsed && <div className="min-w-0"><p className="technical-label text-primary">clinical workspace</p><p className="truncate text-sm font-bold tracking-tight">Retina Signal</p></div>}
            </div>
          </SidebarHeader>
          <SidebarContent className="gap-0 px-2"><p className="technical-label px-2 pb-2 pt-4 text-muted-foreground group-data-[collapsible=icon]:hidden">navigation</p><SidebarMenu>{menuItems.map(item => <SidebarMenuItem key={item.path}><SidebarMenuButton isActive={location === item.path} onClick={() => setLocation(item.path)} tooltip={item.label} className="h-11 rounded-lg"><item.icon className="h-4 w-4" /><span>{item.label}</span></SidebarMenuButton></SidebarMenuItem>)}</SidebarMenu></SidebarContent>
          <SidebarFooter className="p-3"><DropdownMenu><DropdownMenuTrigger asChild><button className="flex w-full items-center gap-3 rounded-lg border border-transparent px-1 py-1 text-left transition-colors hover:border-border hover:bg-accent/50 focus:outline-none focus-visible:ring-2 focus-visible:ring-ring"><Avatar className="h-8 w-8 shrink-0 border"><AvatarFallback className="bg-secondary text-xs font-semibold">{user?.name?.charAt(0).toUpperCase() || "U"}</AvatarFallback></Avatar><div className="min-w-0 flex-1 group-data-[collapsible=icon]:hidden"><p className="truncate text-sm font-medium leading-none">{user?.name || "Clinician"}</p><p className="mt-1.5 truncate font-mono text-[10px] text-muted-foreground">MANUS OAUTH</p></div></button></DropdownMenuTrigger><DropdownMenuContent align="end" className="w-48"><DropdownMenuItem onClick={logout} className="cursor-pointer text-destructive focus:text-destructive"><LogOut className="mr-2 h-4 w-4" />Sign out</DropdownMenuItem></DropdownMenuContent></DropdownMenu></SidebarFooter>
        </Sidebar>
        {!isCollapsed && <div className="absolute right-0 top-0 z-50 h-full w-1 cursor-col-resize transition-colors hover:bg-primary/30" onMouseDown={() => setIsResizing(true)} />}
      </div>
      <SidebarInset>
        {isMobile && <div className="no-print sticky top-0 z-40 flex h-14 items-center gap-2 border-b bg-background/95 px-3 backdrop-blur"><SidebarTrigger className="h-9 w-9 rounded-lg border bg-background" /><div><p className="technical-label text-primary">Retina Signal</p><p className="text-sm font-semibold">{active}</p></div></div>}
        <main className="min-h-screen flex-1 p-4 sm:p-6 lg:p-8">{children}</main>
      </SidebarInset>
    </>
  );
}
