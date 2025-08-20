import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "VVL.DynamicBatchAnythingUI",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (!nodeData) return;
		const isMatch = nodeData.name === "VVL Dynamic Batch Anything" || nodeData.name === "DynamicBatchAnything";
		if (!isMatch) return;

		// 自动管理输入端口：始终保证末尾有一个空输入；移除末尾未使用的多余输入
		function removeUnusedInputsFromEnd(node, minInputs = 2) {
			if (!node.inputs) return;
			while (node.inputs.length > minInputs) {
				const last = node.inputs[node.inputs.length - 1];
				if (!last || last.link == null) {
					node.removeInput(node.inputs.length - 1);
				} else {
					break;
				}
			}
		}

        function ensureTrailingEmpty(node) {
            if (!node.inputs) node.inputs = [];
            const last = node.inputs[node.inputs.length - 1];
            if (!last || last.link != null) {
                const nextIndex = node.inputs.length + 1;
                node.addInput(`input_${nextIndex}`, node._type || "*");
            }
        }

		nodeType.prototype.onNodeCreated = function () {
			this._type = "*";
			// 初始化两个输入端口
			if (!this.inputs) this.inputs = [];
			if (this.inputs.length === 0) {
				this.addInput("input_1", this._type);
				this.addInput("input_2", this._type);
			}
			this.scheduleStabilize?.();
		};

		nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, linkInfo, ioSlot) {
			if (typeof this._debounceTimer !== "undefined") clearTimeout(this._debounceTimer);
			this._debounceTimer = setTimeout(() => this.stabilize?.(), 64);
		};

		nodeType.prototype.onConnectionsChainChange = function () {
			if (typeof this._debounceTimer !== "undefined") clearTimeout(this._debounceTimer);
			this._debounceTimer = setTimeout(() => this.stabilize?.(), 64);
		};

		nodeType.prototype.stabilize = function () {
			removeUnusedInputsFromEnd(this, 2);
			ensureTrailingEmpty(this);
			// 更新输出端口类型（维持为 *）
			if (this.outputs && this.outputs[0]) {
				this.outputs[0].type = this._type || "*";
				this.outputs[0].label = "*";
			}
		};
	},
});

