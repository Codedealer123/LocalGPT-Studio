<script>
  import '@awesome.me/webawesome/dist/styles/webawesome.css';
  import '@awesome.me/webawesome/dist/components/icon/icon.js';

    
  export let isSidebarOpen;
  export let isMobile;
  export let toggleSidebar;
  export let name = "";

  import { customPrompt } from "./ts/customPrompt";

  const menuItems = [
    { icon: 'comment-medical', label: 'New chat', badge: false },
    { icon: 'comment-dots', label: 'Chats', badge: false },
    { icon: 'sd-card', label: 'Artifacts', badge: false },
  ];

  // State to manage the account settings dropdown
  let isMenuOpen = false;

  function toggleMenu(event) {
    event.stopPropagation(); // Prevents instant closing from event bubbling
    isMenuOpen = !isMenuOpen;
  }

  function closeMenu() {
    isMenuOpen = false;
  }

  const changeUsername = async () => {
    name = await customPrompt({title: 'Change Username', message: "", type: "text", placeholder: "John Smith"});
  }
</script>

<svelte:window on:click={closeMenu} />

{#if isMobile && isSidebarOpen}
  <button class="sidebar-backdrop" on:click={toggleSidebar} aria-label="Close menu"></button>
{/if}

<aside class="sidebar" data-state={isSidebarOpen ? 'open' : 'closed'}>
  <div class="sidebar-content-wrapper">
    <div class="sidebar-top">
      <div class="sidebar-header">
        <span class="brand-logo">LocalGPT</span>
        <button class="icon-btn" on:click={toggleSidebar} aria-label="Collapse panel">
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/></svg>
        </button>
      </div>

      <nav class="nav-list">
        {#each menuItems as item}
          <button class="nav-item">
            <div class="nav-item-left">
              <wa-icon name={item.icon}></wa-icon>
              <span>{item.label}</span>
            </div>
            {#if item.badge}
              <span class="badge">{item.badge}</span>
            {/if}
          </button>
        {/each}
      </nav>

      <div class="recents-divider">
        <span>Recents</span>
        <span class="sort-icon">↕</span>
      </div>
    </div>

    <div class="sidebar-footer-container">
      {#if isMenuOpen}
        <!-- svelte-ignore a11y_click_events_have_key_events -->
        <!-- svelte-ignore a11y_no_static_element_interactions -->
        <div class="account-popup" on:click|stopPropagation>

          <button class="popup-item logout" on:click={changeUsername}>
            <div class="popup-item-left">
              <span class="popup-icon">↱</span>
              <span>Change name</span>
            </div>
          </button>

          <button class="popup-item">
            <div class="popup-item-left">
              <span class="popup-icon">⚙️</span>
              <span>Settings</span>
            </div>
          </button>
          
        </div>
      {/if}

      <button class="sidebar-footer" on:click={toggleMenu} aria-expanded={isMenuOpen}>
        <div class="footer-left">
          <div class="avatar">M</div>
          <div class="user-meta">
            <span class="username">{name}</span>
          </div>
        </div>
      </button>
    </div>
  </div>
</aside>

<style>
  .sidebar-backdrop {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 40;
    background-color: rgba(0, 0, 0, 0.6);
    border: none;
    cursor: pointer;
    padding: 0;
    animation: fadeIn 0.2s ease-in-out;
  }

  @keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .sidebar {
    position: fixed;
    top: 0;
    bottom: 0;
    left: 0;
    z-index: 50;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    width: var(--sidebar-width);
    background-color: var(--bg-main);
    border-right: 1px solid var(--border-color);
    padding: 12px;
    box-sizing: border-box;
    transform: translateX(-320px); 
    visibility: hidden;
    transition: transform 0.25s ease-in-out, 
                visibility 0.25s ease-in-out,
                width 0.25s ease-in-out,
                padding 0.25s ease-in-out,
                border-color 0.25s ease-in-out;

    &[data-state="open"] {
      transform: translateX(0);
      visibility: visible;
    }
  }

  .sidebar-content-wrapper {
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
    width: calc(var(--sidebar-width) - 24px); 
    flex-shrink: 0;
    opacity: 1;
    transition: opacity 0.12s ease-in-out; 
  }

  .sidebar[data-state="closed"] .sidebar-content-wrapper {
    opacity: 0;
    pointer-events: none;
  }

  .sidebar-header {
    font-family: "Nunito", sans-serif;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 8px;
    margin-bottom: 16px;
  }

  .brand-logo {
    font-family: var(--font-serif);
    font-size: 20px;
    font-weight: 600;
    color: var(--brand-accent);
  }

  .icon-btn {
    background: none;
    border: none;
    padding: 6px;
    border-radius: 4px;
    color: var(--text-muted);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .icon-btn:hover {
    background-color: var(--bg-surface);
    color: var(--text-main);
  }

  .nav-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .nav-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    background: none;
    border: none;
    padding: 8px 12px;
    border-radius: 8px;
    color: var(--text-main);
    font-size: 14px;
    cursor: pointer;
    text-align: left;
    box-sizing: border-box;
  }

  .nav-item:hover {
    background-color: var(--bg-surface);
  }

  .nav-item-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .nav-icon {
    opacity: 0.8;
    font-size: 16px;
  }

  .badge {
    font-size: 10px;
    color: var(--text-muted);
    background-color: var(--bg-surface-light);
    border: 1px solid #3E3E3E;
    padding: 2px 6px;
    border-radius: 4px;
  }

  .recents-divider {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: var(--text-muted);
    margin-top: 24px;
    padding: 0 12px;
  }

  /* --- Account Floating Popup Menu Styles --- */
  .sidebar-footer-container {
    position: relative;
    width: 100%;
  }

  .account-popup {
    position: absolute;
    bottom: calc(100% + 8px);
    left: 0;
    width: 100%;
    background-color: #262626; /* Deep charcoal surface card matches screenshot */
    border: 1px solid #333333;
    border-radius: 12px;
    padding: 8px;
    box-sizing: border-box;
    z-index: 60;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.5);
    animation: popupFadeIn 0.15s ease-out;
  }

  @keyframes popupFadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .popup-email {
    font-size: 13px;
    color: var(--text-muted);
    padding: 6px 10px 10px 10px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .popup-divider {
    height: 1px;
    background-color: #333333;
    margin: 6px 0;
  }

  .popup-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    background: none;
    border: none;
    border-radius: 6px;
    padding: 8px 10px;
    color: var(--text-main);
    font-size: 14px;
    cursor: pointer;
    text-align: left;
    box-sizing: border-box;
  }

  .popup-item:hover {
    background-color: #2F2F2F;
  }

  .popup-item-left {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .popup-icon {
    font-size: 15px;
    opacity: 0.85;
    display: inline-flex;
    width: 18px;
    justify-content: center;
  }

  .popup-shortcut {
    font-size: 11px;
    color: #666666;
  }

  .popup-arrow {
    color: var(--text-muted);
    font-size: 16px;
  }

  /* --- Interactive Sidebar Footer Toggle Button --- */
  .sidebar-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    background: none;
    border: none;
    border-top: 1px solid var(--border-color);
    padding: 12px 6px 4px 6px;
    cursor: pointer;
    text-align: left;
    box-sizing: border-box;
  }

  .sidebar-footer:hover .username {
    color: #ffffff;
  }

  .footer-left {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .user-meta {
    display: flex;
    flex-direction: column;
  }

  .avatar {
    width: 32px;
    height: 32px;
    background-color: #D2C4B7;
    color: #191919;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    font-weight: 700;
  }

  .username {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-main);
    line-height: 1.2;
  }

  .plan-label {
    font-size: 12px;
    color: var(--text-muted);
  }

  .footer-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    color: var(--text-muted);
  }

  .download-icon {
    font-size: 14px;
    padding: 4px;
    border-radius: 6px;
    border: 1px solid #2A2A2A;
    background-color: #1E1E1E;
  }

  .download-icon:hover {
    color: var(--text-main);
    background-color: var(--bg-surface-light);
  }

  .selector-arrows {
    font-size: 12px;
  }

  @media (min-width: 768px) {
    .sidebar {
      position: relative;
      transform: none !important;
      visibility: visible !important;
      flex-shrink: 0;
      
      &[data-state="closed"] {
        width: 0px;
        padding-left: 0px;
        padding-right: 0px;
        border-right-color: transparent;
        overflow: hidden;
      }
    }
  }
</style>