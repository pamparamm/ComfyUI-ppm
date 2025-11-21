const { app } = window.comfyAPI.app;

app.registerExtension({
    name: "MaskCompositePPM",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "MaskCompositePPM") {
            const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info, _) {
                const mask_input_name = "mask_";

                const masks = this.inputs
                    .filter(input => input.name.startsWith(mask_input_name));

                // Remove last unused inputs (skipping mask_1)
                while (
                    (masks.length > 1)
                    && (masks.at(-1).link == null)
                ) {
                    masks.pop();
                    this.removeInput(this.inputs.length - 1);
                }

                // Add new 'mask' inputs if necessary
                if (masks.at(-1)?.link != null) {
                    const slot_id = masks.length + 1;
                    this.addInput(`${mask_input_name}${slot_id}`, "MASK", { optional: true });
                }

                return origOnConnectionsChange?.apply(this, arguments);
            };
        }
    },
});
