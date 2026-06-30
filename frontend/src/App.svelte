<script>
  import Sidebar from './lib/Sidebar.svelte';
  import MainChat from './lib/Chat.svelte';
  import { chatManager } from './lib/ts/chatManager.svelte';

  let innerWidth = $state(1024);
  let isSidebarOpen = $state(true);
  let hasAutoCollapsed = $state(false);
  let name = $state("Guest");

  let isMobile = $derived(innerWidth < 768);

  // GLOBAL CHAT MENU STATE (key addition)
  let openChatMenuId = $derived(chatManager.state.openChatMenuId);

  $effect(() => {
    if (isMobile && !hasAutoCollapsed) {
      isSidebarOpen = false;
      hasAutoCollapsed = true;
    } else if (!isMobile && hasAutoCollapsed) {
      isSidebarOpen = true;
      hasAutoCollapsed = false;
    }
  });

  function toggleSidebar() {
    isSidebarOpen = !isSidebarOpen;
  }

  // close menu on outside click
  function closeChatMenu() {
    chatManager.closeChatMenu();
  }

  let menuPos = $derived(chatManager.menuPos);
</script>

<svelte:window bind:innerWidth />

<div class="app-container">

  <Sidebar
    {isSidebarOpen}
    {isMobile}
    {toggleSidebar}
    {name}
  />

  <MainChat
    {isSidebarOpen}
    {toggleSidebar}
  />
</div>

<style>
  /* Global Design Variables */
  :global(:root) {
    --bg-main: #191919;
    --bg-surface: #222222;
    --bg-surface-light: #2A2A2A;
    --bg-pill: #1E1E1E;
    --bg-hover: #252525;
    
    --text-main: #E3E3E3;
    --text-muted: #8E8E8E;
    --text-placeholder: #555555;
    --brand-accent: #D97753;
    --brand-fallback: #CC6A47;
    --title-color: #D2C4B7;
    
    --border-color: #2A2A2A;
    --border-color-rgb: 42, 42, 42;
    --border-focus: #444444;
    
    --font-sans: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-serif: Georgia, Cambria, "Times New Roman", Times, serif;
    --sidebar-width: 260px;
  }

  :global(body) {
    margin: 0;
    padding: 0;
    background-color: var(--bg-main);
    color: var(--text-main);
    font-family: var(--font-sans);
    -webkit-font-smoothing: antialiased;
    overflow: hidden;
  }

  .app-container {
    display: flex;
    width: 100vw;
    height: 100vh;
    overflow: hidden;
    position: relative;
  }
    
  :global(#app), :global(#root) {
    overflow: hidden;
    width: 100%;
    height: 100%;
  }

  .chat-context-menu {
    position: fixed;
    z-index: 9999;
  
    background: var(--bg-surface);
    border: 1px solid var(--border-color);
    border-radius: 12px;
  
    min-width: 180px;
    padding: 6px;
  
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
  
    animation: menuIn 0.12s ease-out;
  
    transform-origin: top left;
    transition: top 0.08s ease-out, left 0.08s ease-out, opacity 0.12s ease;
  }
  
  .menu-item {
    width: 100%;
    background: none;
    border: none;
    color: var(--text-main);
  
    padding: 8px 10px;
    text-align: left;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
  }
  
  .menu-item:hover {
    background: var(--bg-surface-light);
  }
  
  .menu-item.danger {
    color: #ff5c5c;
  }
  
  .menu-item.danger:hover {
    background: rgba(255, 92, 92, 0.1);
  }
  
  @keyframes menuIn {
    from {
      opacity: 0;
      transform: scale(0.96);
    }
    to {
      opacity: 1;
      transform: scale(1);
    }
  }
</style>