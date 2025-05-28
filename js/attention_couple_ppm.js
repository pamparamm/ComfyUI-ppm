import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AttentionCouplePPM",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AttentionCouplePPM") {
            const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (_type, _index, _connected, _link_info, _) {
                const cond_input_name = "cond_";
                const mask_input_name = "mask_";

                const base_cond = this.inputs
                    ?.filter(input => input.name === "base_cond")[0];
                const base_mask = this.inputs
                    ?.filter(input => input.name === "base_mask")[0];

                const conds = this.inputs
                    ?.filter(input => input.name.startsWith(cond_input_name));
                const masks = this.inputs
                    ?.filter(input => input.name.startsWith(mask_input_name));

                // Remove last unused inputs (skipping cond_1 and mask_1)
                while (
                    (conds.length > 0 && masks.length > 0)
                    && (conds.at(-1).link == null && masks.at(-1).link == null)
                ) {
                    masks.pop();
                    this.removeInput(this.inputs.length - 1);
                    conds.pop();
                    this.removeInput(this.inputs.length - 1);
                }

                const is_full = conds.at(-1)?.link != null && masks.at(-1)?.link != null;
                const is_only_base = conds.length === 0 && masks.length === 0 && base_cond.link != null && base_mask.link != null;

                // Add new 'cond' and 'mask' inputs if necessary
                if (is_only_base || is_full) {
                    const slot_id = conds.length + 1;
                    this.addInput(`${cond_input_name}${slot_id}`, "CONDITIONING", { optional: true });
                    this.addInput(`${mask_input_name}${slot_id}`, "MASK", { optional: true });
                }

                return origOnConnectionsChange?.apply(this, arguments);
            };
        }
    },
});
