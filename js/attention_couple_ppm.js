import { app } from "/scripts/app.js";

app.registerExtension({
    name: "AttentionCouplePPM",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AttentionCouplePPM") {
            const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info, _) {
                const cond_input_name = "cond_";
                const mask_input_name = "mask_";

                const conds = this.inputs
                    .filter(input => input.name.startsWith(cond_input_name));
                const masks = this.inputs
                    .filter(input => input.name.startsWith(mask_input_name));

                // Remove last unused inputs (skipping cond_1 and mask_1)
                while (
                    (conds.length > 1 && masks.length > 1)
                    && (conds.at(-1).link == null && masks.at(-1).link == null)
                ) {
                    conds.pop();
                    this.removeInput(this.inputs.length - 1);
                    masks.pop();
                    this.removeInput(this.inputs.length - 1);
                }

                // Add new 'cond' and 'mask' inputs if necessary
                if (conds.at(-1)?.link != null || masks.at(-1)?.link != null) {
                    const slot_id = conds.length + 1;
                    this.addInput(`${cond_input_name}${slot_id}`, "CONDITIONING", { optional: true });
                    this.addInput(`${mask_input_name}${slot_id}`, "MASK", { optional: true });
                }

                return origOnConnectionsChange?.apply(this, arguments);
            };
        }
    },
});

