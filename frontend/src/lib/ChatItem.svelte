<script lang="ts">
  import { chatManager } from "./ts/chatManager.svelte";
  import { customPrompt } from "./ts/customPrompt";
  import { onMount } from "svelte";
  
  function handleClickOutside(e: MouseEvent) {
    if (open) open = false;
  }
  
  onMount(() => {
    window.addEventListener("click", handleClickOutside);
    return () => window.removeEventListener("click", handleClickOutside);
  });

  let { chat, activeChatId } = $props();

  let open = $state(false);

  function toggleMenu(e: MouseEvent) {
    e.stopPropagation();
    open = !open;
  }

  function closeMenu() {
    open = false;
  }

  async function rename() {
    const title = await customPrompt({
      title: "Rename chat",
      type: "text",
      placeholder: chat.title
    });

    if (typeof title === "string" && title.trim()) {
      chatManager.renameChat(chat.id, title.trim());
    }

    chatManager.persist();

    closeMenu();
  }

  function remove() {
    chatManager.deleteChat(chat.id);
    chatManager.persist();
    closeMenu();
  }
</script>

<div class="chat-item-wrapper" >
  <button
    class="chat-item"
    class:active={chat.id === activeChatId}
    on:click={() => chatManager.setActiveChat(chat.id)}
  >
    <span class="chat-title">{chat.title}</span>
  </button>

  <button class="chat-menu-btn" on:click={toggleMenu}>
    ⋯
  </button>

  {#if open}
    <div class="chat-menu" on:click|stopPropagation>
      <button class="chat-menu-item" on:click={rename}>
        Rename
      </button>

      <button class="chat-menu-item danger" on:click={remove}>
        Delete
      </button>
    </div>
  {/if}
</div>

<style>
  .chat-item {
    width: 100%;
    background: none;
    border: none;
    color: var(--text-main);
    text-align: left;
    padding: 8px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    margin-top: 2px;
  }
  
  .chat-item:hover {
    background: var(--bg-surface);
  }
  
  .chat-item.active {
    background: var(--bg-surface);
    font-weight: 600;
  }
  
  .chat-title {
    display: block;
          overflow: hidden;
          white-space: nowrap;
          text-overflow: ellipsis;
        }
        
        .chat-item-wrapper {
          position: relative;
          display: flex;
          align-items: center;
          gap: 6px;
          border-radius: 8px;
        }
        
        /* hide by default */
        .chat-menu-btn {
          opacity: 0;
          pointer-events: none;
          background: none;
          border: none;
          color: var(--text-muted);
          cursor: pointer;
          padding: 4px 6px;
          border-radius: 6px;
          transition: opacity 0.15s ease;
        }
        
        .chat-menu {
          position: fixed;
          min-width: 160px;
        
          background: #1f1f1f;
          border: 1px solid #2a2a2a;
          border-radius: 10px;
          padding: 4px;
        
          box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
          z-index: 9999;
        
          animation: menuIn 0.12s ease-out;
        }
    
        .chat-menu-item {
          width: 100%;
          text-align: left;
          background: none;
          border: none;
        
          padding: 6px 8px;
          border-radius: 6px;
          font-size: 13px;
          color: var(--text-main);
          cursor: pointer;
        }
    
        .chat-menu-item:hover {
          background: var(--bg-surface);
        }
        
        .chat-menu-item.danger {
          color: #ff5a5a;
        }
        
        /* show on hover */
        .chat-item-wrapper:hover .chat-menu-btn {
          opacity: 1;
          pointer-events: auto;
        }
        
        .chat-item-wrapper:hover .chat-item {
          background: var(--bg-surface);
        }
        
        .chat-item-wrapper:hover .chat-menu-btn {
          opacity: 1;
        }
</style>