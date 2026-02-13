/**
 * MTI-EVO UI Effects Module
 * Handles visual embellishments: Glitch text, typing effects, and particle systems.
 */

export class UIEffects {
    constructor() {
        this.glitchIntervals = new Map();
    }

    /**
     * Applies a cyberpunk glitch effect to a text element
     * @param {HTMLElement} element 
     * @param {string} finalText 
     */
    glitchText(element, finalText) {
        if (!element) return;

        const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?';
        let iterations = 0;

        if (this.glitchIntervals.has(element)) {
            clearInterval(this.glitchIntervals.get(element));
        }

        const interval = setInterval(() => {
            element.innerText = finalText
                .split('')
                .map((char, index) => {
                    if (index < iterations) {
                        return finalText[index];
                    }
                    return chars[Math.floor(Math.random() * chars.length)];
                })
                .join('');

            if (iterations >= finalText.length) {
                clearInterval(interval);
                this.glitchIntervals.delete(element);
            }

            iterations += 1 / 3;
        }, 30);

        this.glitchIntervals.set(element, interval);
    }

    /**
     * Adds a "scanning" class to an element temporarily
     */
    triggerScanEffect(element) {
        element.classList.add('scanning');
        setTimeout(() => element.classList.remove('scanning'), 1000);
    }
}

export const uiEffects = new UIEffects();
