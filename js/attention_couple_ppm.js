// Taken from https://github.com/laksjdjf/cgem156-ComfyUI/blob/1f5533f7f31345bafe4b833cbee15a3c4ad74167/js/attention_couple.js
import { app } from "/scripts/app.js";

app.registerExtension({
    name: "AttentionCouplePPM",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AttentionCouplePPM") {
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function (_, options) {
                const r = origGetExtraMenuOptions?.apply?.(this, arguments);
                options.unshift(
                    {
                        content: "Add Region",
                        callback: () => {
                            var index = 1;
                            if (this.inputs != undefined) {
                                index += this.inputs.length;
                            }
                            this.addInput("cond_" + Math.floor(index / 2), "CONDITIONING");
                            this.addInput("mask_" + Math.floor(index / 2), "MASK");
                        },
                    },
                    {
                        content: "Remove Region",
                        callback: () => {
                            if (this.inputs != undefined && this.inputs.at(-2)["type"] === "CONDITIONING") {
                                this.removeInput(this.inputs.length - 1);
                                this.removeInput(this.inputs.length - 1);
                            }
                        },
                    },
                );
                return r;

            }

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
			nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                let cond_input_name = "cond_";
                let mask_input_name = "mask_";
                let slot_i = 1;

                // Count existing input slots
                for (let i = 0; i < this.inputs.length; i++) {
                    let input_i = this.inputs[i];
                    if (input_i.name.startsWith(mask_input_name)) {
                        slot_i++;
                    }
                }

                let last_slot = this.inputs[this.inputs.length - 1];
                let second_last_slot = this.inputs[this.inputs.length - 2];

                // Check if the last two slots are connected
                if (
                    (second_last_slot.link != undefined)
                    && (last_slot.link != undefined)
                ) {
                    // Add new 'cond' and 'mask' slots
                    this.addInput(`${cond_input_name}${slot_i}`, "CONDITIONING");
                    this.addInput(`${mask_input_name}${slot_i}`, "MASK");
                }
            }
        }
    },
});
