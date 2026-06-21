<script lang="ts">
  let { isSidebarOpen, toggleSidebar } = $props();

  import { modelState, changeChosenModel } from "./ts/modelManager.svelte";
  import { markdownToHtml } from "./ts/markdown";
  import { chatManager } from "./ts/chatManager.svelte";

  import {
    initChatController,
    sendChatMessage
  } from "./ts/chatController";

  let searchQuery = $state("");
  let inputMessage = $state("");

  // start websocket + handlers (moved out of component)
  initChatController();

  // ensure chat exists
  if (!chatManager.getActiveChat()) {
    chatManager.createChat("New Chat");
  }

  // reactive messages
  let messages = $derived(
    chatManager.getActiveChat()?.messages ?? []
  );

  // model filter
  let filteredModels = $derived(
    modelState.downloadedModels.filter(model =>
      model.toLowerCase().includes(searchQuery.toLowerCase())
    )
  );

  function handleSend() {
    if (!inputMessage.trim()) return;

    sendChatMessage(inputMessage);
    inputMessage = "";
  }

  function handleKeyDown(event: KeyboardEvent) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSend();
    }
  }
</script>

<main class="chat-main">
  <header class="chat-header">
    <button class="menu-toggle-btn" class:hidden-on-desktop={isSidebarOpen} onclick={toggleSidebar} aria-label="Toggle structural access menu column panel layout visibility status control">
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/></svg>
    </button>
  </header>

  <div class="workspace-center">
    {#if messages.length === 0}
      <div class="splash-container">
        <h1 class="welcome-title">
          <span class="welcome-star">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="#D87758" xmlns="http://www.w3.org/2000/svg"><g stroke-width="0"/><g stroke-linecap="round" stroke-linejoin="round"/><g fill-rule="evenodd" clip-rule="evenodd"><path d="M11.994 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7m0-2.006a1.494 1.494 0 1 1 0-2.988 1.494 1.494 0 0 1 0 2.988"/><path d="M12 5C7.189 5 3.917 7.609 2.19 9.48a3.68 3.68 0 0 0 0 5.04C3.916 16.391 7.188 19 12 19s8.083-2.609 9.81-4.48a3.68 3.68 0 0 0 0-5.04C20.084 7.609 16.812 5 12 5m-8.341 5.837C5.189 9.18 7.967 7 12 7s6.812 2.18 8.341 3.837a1.68 1.68 0 0 1 0 2.326C18.811 14.82 16.033 17 12 17s-6.812-2.18-8.341-3.837a1.68 1.68 0 0 1 0-2.326"/></g></svg>
          </span> Welcome, Mukilan
        </h1>
      </div>
    {:else}
      <div class="conversation-container">
        {#each messages as msg}
          <div class="message-row" data-role={msg.role}>
            <div class="message-text">
              {@html markdownToHtml(msg.content)}
            </div>
          </div>
        {/each}
      </div>
    {/if}

    <div class="bottom-controls-wrapper">
      <div class="input-console-box">
        <textarea 
          placeholder="How can I help you today?" 
          class="chat-textarea"
          bind:value={inputMessage}
          onkeydown={handleKeyDown}
        ></textarea>
        
        <div class="toolbar-ribbon">
          <button class="attach-btn" aria-label="Attach file button component item media asset option controller">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
          </button>

          <div class="utility-tools">
            <label class="popup">
              <input type="checkbox">
              <div role="button" aria-haspopup="listbox" tabindex="0">
                <span class="truncate-chosen-model">{modelState.chosenModel}</span> 
                <span class="dropdown-arrow">▼</span>
              </div>
              <nav class="popup-window">
                <legend>Models</legend>

                <div class="model-search-container">
                  <span class="search-icon">🔍</span>
                  <input 
                    type="text" 
                    class="model-search-input" 
                    placeholder="Search models..." 
                    bind:value={searchQuery}
                    onclick={(e) => { e.stopPropagation(); e.preventDefault(); }}
                    onmousedown={(e) => e.stopPropagation()}
                    onpointerdown={(e) => e.stopPropagation()}
                  />
                </div>

                <ul>
                  {#each filteredModels as model}
                  <li>
                      <button onclick={()=>changeChosenModel(model)}>
                          <span>{model}</span>
                      </button>
                  </li>
                  {/each}
                  {#if filteredModels.length === 0}
                    <li class="no-results">No matching models</li>
                  {/if}
                </ul>
              </nav>
            </label>
            <button class="sendButton" onclick={handleSend}>
              <svg
                height="20"
                width="20"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path d="M0 0h24v24H0z" fill="none"></path>
                <path
                  d="M5 13c0-5.088 2.903-9.436 7-11.182C16.097 3.564 19 7.912 19 13c0 .823-.076 1.626-.22 2.403l1.94 1.832a.5.5 0 0 1 .095.603l-2.495 4.575a.5.5 0 0 1-.793.114l-2.234-2.234a1 1 0 0 0-.707-.293H9.414a1 1 0 0 0-.707.293l-2.234 2.234a.5.5 0 0 1-.793-.114l-2.495-4.575a.5.5 0 0 1 .095-.603l1.94-1.832C5.077 14.626 5 13.823 5 13zm1.476 6.696l.817-.817A3 3 0 0 1 9.414 18h5.172a3 3 0 0 1 2.121.879l.817.817.982-1.8-1.1-1.04a2 2 0 0 1-.593-1.82c.124-.664.187-1.345.187-2.036 0-3.87-1.995-7.3-5-8.96C8.995 5.7 7 9.13 7 13c0 .691.063(1.372).187 2.037a2 2 0 0 1-.593 1.82l-1.1 1.039.982 1.8zM12 13a2 2 0 1 1 0-4 2 2 0 0 1 0 4z"
                  fill="currentColor"
                ></path>
              </svg>
              <span>SEND</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </div>
</main>

<style>
  .popup {
    --burger-color: var(--text-muted);
    --nav-padding-x: 0.25em;
    --nav-padding-y: 0.625em;
    --nav-border-radius: 8px;
    --nav-border-color: rgba(var(--border-color-rgb), 100);
    --nav-border-width: 1px;
    --nav-shadow-color: rgba(0, 0, 0, 0.3);
    --nav-shadow-width: 0 10px 25px -5px;
    --nav-bg: var(--bg-surface-light);
    --nav-font-family: var(--font-sans, inherit);
    --nav-default-scale: .95;
    --nav-active-scale: 1;
    --nav-position-left: 0;
    --nav-position-right: unset;
    --nav-title-size: 0.625em;
    --nav-title-color: var(--text-muted);
    --nav-title-padding-x: 1rem;
    --nav-title-padding-y: 0.25em;
    --nav-button-padding-x: 1rem;
    --nav-button-padding-y: 0.5em;
    --nav-border-radius: 6px;
    --nav-button-font-size: 12px;
    --nav-button-hover-bg: var(--bg-hover);
    --nav-button-hover-text-color: var(--text-main);
    --nav-button-distance: 0.875em;
    --underline-border-width: 1px;
    --underline-border-color: rgba(var(--border-color-rgb), 100);
    --underline-margin-y: 0.3125em;
  }
  
  .popup {
    display: inline-block;
    position: relative;
  }
  
  .popup input {
    display: none;
  }
  
  .truncate-chosen-model {
    display: inline-block;
    max-width: 120px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    vertical-align: bottom;
  }
  
  .popup > div[role="button"] {
    display: flex;
    align-items: center;
    gap: 4px;
    background-color: var(--bg-surface-light);
    border: none;
    border-radius: 6px;
    padding: 4px 10px;
    color: var(--text-muted);
    font-size: 11px;
    cursor: pointer;
    transition: color 0.2s ease, background-color 0.2s ease;
  }
  
  .popup > div[role="button"]:hover,
  .popup > div[role="button"]:focus-visible {
    color: var(--text-main);
  }
  
  .popup > div[role="button"] * {
    color: inherit;
  }
  
  .popup-window {
    transform: scale(var(--nav-default-scale));
    visibility: hidden;
    opacity: 0;
    position: absolute;
    z-index: 1000;
    padding: var(--nav-padding-y) var(--nav-padding-x);
    background: var(--nav-bg);
    color: var(--text-muted);
    border-radius: var(--nav-border-radius);
    border: var(--nav-border-width) solid var(--nav-border-color);
    box-shadow: var(--nav-shadow-width) var(--nav-shadow-color);
    top: calc(100% + 8px);
    left: var(--nav-position-left);
    right: var(--nav-position-right);
    transition: transform 0.15s ease, opacity 0.15s ease;
    min-width: 220px;
  }
  
  .model-search-container {
    display: flex;
    align-items: center;
    gap: 6px;
    background-color: #191919;
    border: 1px solid var(--border-color);
    border-radius: 6px;
    margin: 4px 6px 10px 6px;
    padding: 4px 8px;
  }

  .search-icon {
    font-size: 11px;
    opacity: 0.5;
  }

  .model-search-input {
    background: none;
    border: none;
    outline: none;
    width: 100%;
    color: var(--text-main);
    font-family: var(--font-sans);
    font-size: 12px;
  }

  .no-results {
    padding: 6px 1rem;
    font-size: 12px;
    color: var(--text-placeholder);
    text-align: center;
  }
  
  .popup-window,
  .popup-window * {
    color: inherit;
    box-sizing: border-box;
  }
  
  .popup-window legend {
    padding: var(--nav-title-padding-y) var(--nav-title-padding-x);
    margin: 0;
    color: var(--nav-title-color);
    font-size: var(--nav-title-size);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  
  .popup-window ul {
    margin: 0;
    padding: 0;
    list-style: none;
    max-height: 200px;
    overflow-y: auto;
  }
  
  .popup-window ul li {
    margin: 0;
    padding: 0;
  }
  
  .popup-window ul button {
    width: 100%;
    border: none;
    outline: 1px solid #1f1f1f;
    display: flex;
    align-items: center;
    background: transparent;
    color: var(--text-muted);
    font-size: var(--nav-button-font-size);
    padding: var(--nav-button-padding-y) var(--nav-button-padding-x);
    border-radius: var(--nav-button-border-radius);
    cursor: pointer;
    column-gap: var(--nav-button-distance);
    transition: background-color 0.15s ease, color 0.15s ease;
  }
  
  .popup-window ul button span {
    color: inherit;
  }
  
  .popup-window ul button:hover,
  .popup-window ul button:focus-visible {
    background: var(--nav-button-hover-bg);
    color: var(--nav-button-hover-text-color);
  }
  
  .popup-window hr {
    margin: var(--underline-margin-y) 0;
    border: none;
    border-bottom: var(--underline-border-width) solid var(--underline-border-color);
  }
  
  .popup input:checked ~ nav {
    transform: scale(var(--nav-active-scale));
    visibility: visible;
    opacity: 1;
  }
  
  .sendButton {
    display: flex;
    align-items: center;
    font-family: inherit;
    cursor: pointer;
    font-weight: 500;
    font-size: 15px;
    padding: 0.8em 1.3em 0.8em 0.9em;
    color: 1f1f1f;
    background: #2E2E2E;
    border: none;
    letter-spacing: 0.05em;
    border-radius: 16px;
  }

  .sendButton svg {
    margin-right: 3px;
    transform: rotate(30deg);
    transition: transform 0.5s cubic-bezier(0.76, 0, 0.24, 1);
  }

  .sendButton span {
    transition: transform 0.5s cubic-bezier(0.76, 0, 0.24, 1);
  }

  .sendButton:hover svg {
    transform: translateX(5px) rotate(90deg);
  }

  .sendButton:hover span {
    transform: translateX(7px);
  }
  
  /* Layout Architecture: Anchors chat view boundaries natively */
  .chat-main {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    background-color: var(--bg-main);
    padding: 12px;
    box-sizing: border-box;
    height: 100vh; /* Keeps app frame strictly matching screen heights */
    width: 100%;
    overflow: hidden; /* Stops body duplicate scrolling bars */
  }

  .chat-header {
    display: flex;
    width: 100%;
    align-items: center;
    justify-content: space-between;
    min-height: 40px;
    flex-shrink: 0;
  }

  .menu-toggle-btn {
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

  .menu-toggle-btn:hover {
    background-color: var(--bg-surface);
    color: var(--text-main);
  }

  /* Full vertical framework stretching down on all formats */
  .workspace-center {
    width: 100%;
    max-width: 672px;
    display: flex;
    flex-direction: column;
    flex: 1;
    overflow: hidden; /* Protects input row baseline positioning */
    position: relative;
  }

  /* Centers the brand header nicely when history is clean */
  .splash-container {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* Responsive Text Feed Flow (Locks scroll boundaries cleanly above inputs) */
  .conversation-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    width: 100%;
    overflow-y: auto;
    padding-right: 4px;
    gap: 16px;
    box-sizing: border-box;
    scrollbar-width: none;
    -ms-overflow-style: none;
  }

  .conversation-container::-webkit-scrollbar {
    display: none;
  }

  .message-row {
    display: flex;
    width: 100%;
    box-sizing: border-box;
    flex-shrink: 0;
  }

  .message-row[data-role="user"] {
    justify-content: flex-end;
  }

  .message-row[data-role="user"] .message-text {
    background-color: var(--bg-surface-light, #2a2a2a);
    border: 1px solid rgba(var(--border-color-rgb), 0.5);
    border-radius: 16px;
    padding: 10px 16px;
    max-width: 85%;
    text-align: left;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  }

  .message-row[data-role="assistant"] {
    justify-content: flex-start;
    padding: 8px 0;
    border-bottom: 1px solid rgba(var(--border-color-rgb), 0.2);
  }

  .message-row[data-role="assistant"] .message-text {
    background: transparent;
    padding: 0;
    border: none;
    max-width: 100%;
    text-align: left;
  }

  .message-text {
    font-family: var(--font-sans);
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-main);
    white-space: pre-wrap;
  }

  /* Bottom Controls: Clamps console directly to footer limits */
  .bottom-controls-wrapper {
    width: 100%;
    padding: 12px 0;
    background-color: var(--bg-main);
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .welcome-title {
    font-family: var(--font-serif);
    font-size: 24px;
    font-weight: 400;
    color: var(--title-color);
    letter-spacing: 0.05em;
    display: flex;
    align-items: center; 
    justify-content: center;
    gap: 12px;
    margin: 0;
  }
  
  .welcome-star {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
  }
  
  .welcome-star svg {
    width: 100%;
    height: 100%;
    display: block;
  }

  .input-console-box {
    width: 100%;
    background-color: var(--bg-surface);
    border: 1px solid rgba(var(--border-color-rgb), 100);
    border-radius: 16px;
    padding: 12px;
    box-sizing: border-box;
    text-align: left;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    transition: border-color 0.2s;
  }

  .input-console-box:focus-within {
    border-color: var(--border-focus);
  }

  .chat-textarea {
    width: 100%;
    background: none;
    border: none;
    outline: none;
    resize: none;
    font-family: var(--font-sans);
    font-size: 14px;
    color: var(--text-main);
    height: 56px;
  }

  .chat-textarea::placeholder {
    color: var(--text-placeholder);
  }

  .toolbar-ribbon {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 6px;
    padding-top: 6px;
    border-top: 1px solid rgba(42, 42, 42, 0.4);
  }

  .attach-btn {
    background: none;
    border: none;
    border-radius: 50%;
    padding: 8px;
    color: var(--text-muted);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .attach-btn:hover {
    background-color: var(--bg-surface-light);
    color: var(--text-main);
  }

  .utility-tools {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .pills-container {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 6px;
    width: 100%;
  }

  .suggestion-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background-color: var(--bg-pill);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 6px 10px;
    font-size: 11px;
    color: var(--text-muted);
    cursor: pointer;
    transition: background-color 0.2s, color 0.2s;
  }

  .suggestion-pill:hover {
    background-color: var(--bg-hover);
    color: #C5C5C5;
  }

  .pill-prefix {
    opacity: 0.6;
    font-size: 10px;
  }

  /* Responsive Adaptations */
  @media (min-width: 640px) {
    .chat-main { padding: 16px; }
    .welcome-title { font-size: 30px; gap: 12px; }
    .chat-textarea { font-size: 16px; height: 72px; }
    .message-text { font-size: 15px; }
    .suggestion-pill { font-size: 12px; padding: 6px 12px; }
    .pill-prefix { font-size: 11px; }
    .truncate-chosen-model { max-width: 220px; }
  }

  @media (min-width: 768px) {
    .chat-main { padding: 20px 24px 12px 24px; }
    .welcome-title { font-size: 36px; }
    .menu-toggle-btn.hidden-on-desktop {
      opacity: 0;
      pointer-events: none;
    }
  }
</style>