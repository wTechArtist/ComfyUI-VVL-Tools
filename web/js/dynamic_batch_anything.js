import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "VVL.DynamicBatchAnythingUI",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (!nodeData || nodeData.name !== "DynamicBatchAnything") return;

		nodeType.prototype.onNodeCreated = function () {
			this._type = "*";
			this.addWidget("button", "Update inputs", null, () => {
				if (!this.inputs) this.inputs = [];
				const widget = this.widgets?.find((w) => w.name === "inputcount");
				let target = Number(widget?.value ?? 2);
				if (!Number.isFinite(target)) target = 2;
				target = Math.max(1, Math.min(1000, target));

				const numInputs = this.inputs.filter((inp) => inp.type === this._type).length;
				if (target === numInputs) return;

				if (target < numInputs) {
					const toRemove = numInputs - target;
					for (let i = 0; i < toRemove; i++) this.removeInput(this.inputs.length - 1);
				} else {
					for (let i = numInputs + 1; i <= target; i++) this.addInput(`input_${i}`, this._type);
				}
			});
		};
	},
});

