<script>
  import Sidebar from './lib/Sidebar.svelte';
  import MainChat from './lib/Chat.svelte';

  // Svelte 5 State Runes
  let innerWidth = $state(1024);
  let isSidebarOpen = $state(true);
  let hasAutoCollapsed = $state(false);
  let name = $state("Guest");

  // Svelte 5 Derived State (Replaces $: isMobile = ...)
  let isMobile = $derived(innerWidth < 768);

  // Svelte 5 Effect Rune (Replaces $: if (isMobile) ... reactive blocks)
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
</script>

<svelte:window bind:innerWidth />

<div class="app-container">
  <Sidebar {isSidebarOpen} {isMobile} {toggleSidebar} {name} />
  <MainChat {isSidebarOpen} {toggleSidebar} />
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
</style>