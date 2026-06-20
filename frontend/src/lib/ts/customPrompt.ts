/**
 * Triggers a beautiful, awaitable custom modal prompt matching the LocalGPT theme.
 * * @param {Object} params - The configuration parameters.
 * @param {string} params.title - The headline text at the top of the modal.
 * @param {string} params.message - The detailed descriptive body text.
 * @param {'options'|'text'} [params.type='options'] - The prompt mode. 'options' for choice buttons, 'text' for user text input.
 * @param {string} [params.placeholder=''] - Placeholder text used if type is 'text'.
 * @param {string} [params.confirmText='Confirm'] - Label text for the action button.
 * @param {string} [params.cancelText='Cancel'] - Label text for the dismiss button.
 * @param {boolean} [params.isDestructive=false] - If true, styles the action button with a warning/red accent.
 * @returns {Promise<boolean|string|null>} 
 */
export function customPrompt({
  title = "",
  message = "",
  type = 'options',
  placeholder = '',
  confirmText = 'Confirm',
  cancelText = 'Cancel',
  isDestructive = false
}) {
  return new Promise((resolve) => {
    // 1. Create the overlay container
    const overlay = document.createElement('div');
    overlay.className = 'custom-prompt-overlay';

    // 2. Styled text console box matching the MainChat style guidelines
    const inputHTML = type === 'text' 
      ? `<div class="custom-prompt-input-console">
           <input type="text" class="custom-prompt-field" placeholder="${placeholder}" autofocus />
         </div>`
      : '';

    // 3. Setup the inner structural HTML template
    overlay.innerHTML = `
      <div class="custom-prompt-card">
        <h2 class="custom-prompt-title">${title}</h2>
        <p class="custom-prompt-message">${message}</p>
        ${inputHTML}
        <div class="custom-prompt-actions">
          <button class="custom-prompt-btn btn-cancel">${cancelText}</button>
          <button class="custom-prompt-btn btn-confirm ${isDestructive ? 'destructive' : ''}">${confirmText}</button>
        </div>
      </div>
    `;

    // 4. Inject matching application style guidelines down into document head
    if (!document.getElementById('custom-prompt-styles')) {
      const styleSheet = document.createElement('style');
      styleSheet.id = 'custom-prompt-styles';
      styleSheet.textContent = `
        .custom-prompt-overlay {
          position: fixed;
          inset: 0;
          background-color: rgba(0, 0, 0, 0.7);
          display: flex;
          align-items: center;
          justify-content: center;
          z-index: 1000;
          padding: 16px;
          animation: promptFadeIn 0.18s ease-in-out;
        }

        .custom-prompt-card {
          background-color: #222222; /* --bg-surface */
          border: 1px solid #2A2A2A;   /* --border-color */
          border-radius: 16px;         /* Rounded card profile */
          width: 100%;
          max-width: 460px;
          padding: 24px;
          box-sizing: border-box;
          box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
          animation: promptScaleUp 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        }

        .custom-prompt-title {
          font-family: system-ui, -apple-system, sans-serif;
          font-size: 18px;
          font-weight: 500;
          color: #D2C4B7;              /* --title-color */
          margin: 0 0 10px 0;
          letter-spacing: 0.02em;
        }

        .custom-prompt-message {
          font-family: system-ui, -apple-system, sans-serif;
          font-size: 14px;
          line-height: 1.5;
          color: #8E8E8E;              /* --text-muted */
          margin: 0 0 20px 0;
        }

        /* --- Updated Text Input Console Styling --- */
        .custom-prompt-input-console {
          margin-bottom: 24px;
          width: 100%;
          background-color: #191919;   /* --bg-main pill fill */
          border: 1px solid #2A2A2A;   /* --border-color */
          border-radius: 10px;
          padding: 4px 6px;
          box-sizing: border-box;
          transition: border-color 0.2s ease-in-out;
        }

        .custom-prompt-input-console:focus-within {
          border-color: #444444;       /* --border-focus */
        }

        .custom-prompt-field {
          width: 100%;
          background: none;
          border: none;
          outline: none;
          padding: 10px 12px;
          font-family: system-ui, -apple-system, sans-serif;
          font-size: 14px;
          color: #E3E3E3;              /* --text-main */
          box-sizing: border-box;
        }

        .custom-prompt-field::placeholder {
          color: #555555;              /* --text-placeholder */
        }

        /* --- Footer Button System --- */
        .custom-prompt-actions {
          display: flex;
          justify-content: flex-end;
          gap: 10px;
        }

        .custom-prompt-btn {
          font-family: system-ui, -apple-system, sans-serif;
          font-size: 14px;
          font-weight: 500;
          padding: 8px 16px;
          border-radius: 8px;
          cursor: pointer;
          transition: background-color 0.15s, color 0.15s, border-color 0.15s;
          box-sizing: border-box;
        }

        .btn-cancel {
          background: none;
          border: 1px solid #2A2A2A;
          color: #8E8E8E;
        }

        .btn-cancel:hover {
          background-color: #2A2A2A;
          color: #E3E3E3;
        }

        .btn-confirm {
          background-color: #D97753;   /* --brand-accent */
          border: 1px solid #D97753;
          color: #191919;
        }

        .btn-confirm:hover {
          background-color: #e28562;
          border-color: #e28562;
        }

        .btn-confirm.destructive {
          background-color: #B91C1C;
          border-color: #B91C1C;
          color: #ffffff;
        }

        .btn-confirm.destructive:hover {
          background-color: #DC2626;
          border-color: #DC2626;
        }

        @keyframes promptFadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }

        @keyframes promptScaleUp {
          from { transform: scale(0.96); opacity: 0; }
          to { transform: scale(1); opacity: 1; }
        }
      `;
      document.head.appendChild(styleSheet);
    }

    // 5. Cleanup DOM utility helper
    function cleanup(returnValue) {
      overlay.remove();
      resolve(returnValue);
    }

    const inputField = overlay.querySelector('.custom-prompt-field');

    // 6. Action event triggers
    overlay.querySelector('.btn-cancel').addEventListener('click', () => {
      cleanup(type === 'text' ? null : false);
    });

    overlay.querySelector('.btn-confirm').addEventListener('click', () => {
      if (type === 'text') {
        cleanup(inputField ? inputField.value : '');
      } else {
        cleanup(true);
      }
    });

    // Handle pressing 'Enter' key natively inside the text block
    if (inputField) {
      inputField.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
          cleanup(inputField.value);
        } else if (e.key === 'Escape') {
          cleanup(null);
        }
      });
    }

    overlay.addEventListener('click', (e) => {
      if (e.target === overlay) {
        cleanup(type === 'text' ? null : false);
      }
    });

    // 7. Inject node mount point
    document.body.appendChild(overlay);

    // Manual focus backup fallback
    if (inputField) inputField.focus();
  });
}