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
                                if (this.inputs != undefined){
                                    index += this.inputs.length;
                                }
                                this.addInput("cond_" + Math.floor(index / 2), "CONDITIONING");
                                this.addInput("mask_" + Math.floor(index / 2), "MASK");
                            },
                        },
                        {
                            content: "Remove Region",
                            callback: () => {
                                if (this.inputs != undefined && this.inputs.at(-2)["type"] === "CONDITIONING"){
                                    this.removeInput(this.inputs.length - 1);
                                    this.removeInput(this.inputs.length - 1);
                                }
                            },
                        },
                    );
                return r;

            }
        }
    },
});