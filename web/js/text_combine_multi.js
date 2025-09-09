import { app } from "../../../scripts/app.js";

app.registerExtension({
	name: "VVL.TextCombineMulti",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (!nodeData) return;
		
		// 检查是否是TextCombineMulti节点
		const isTextCombine = nodeData.name === "TextCombineMulti";
		
		// 只为TextCombineMulti节点添加调试信息
		if (isTextCombine) {
			console.log("[VVL.TextCombineMulti] Found TextCombineMulti node:", nodeData);
		}
		
		if (isTextCombine) {
				// 动态管理文本widget：根据inputcount添加/删除文本输入框
				function removeExtraTextWidgets(node, maxInputs = 2) {
					if (!node.widgets) return;
					
					// 从后往前移除多余的text widget（但保留控制widget）
					const controlWidgetNames = ["inputcount", "separator", "Update inputs"];
					
					for (let i = node.widgets.length - 1; i >= 0; i--) {
						const widget = node.widgets[i];
						if (widget.name && widget.name.startsWith("text_") && !controlWidgetNames.includes(widget.name)) {
							const num = parseInt(widget.name.replace("text_", ""));
							if (num > maxInputs) {
								node.widgets.splice(i, 1);
							}
						}
					}
				}

				function ensureTextWidgets(node, targetCount) {
					if (!node.widgets) node.widgets = [];
					
					// 找到控制widget的索引（inputcount, separator, Update inputs按钮）
					const controlWidgets = [];
					const controlWidgetNames = ["inputcount", "separator", "Update inputs"];
					
					// 保存控制widget
					for (let i = node.widgets.length - 1; i >= 0; i--) {
						const widget = node.widgets[i];
						if (controlWidgetNames.includes(widget.name)) {
							controlWidgets.unshift(node.widgets.splice(i, 1)[0]);
						}
					}
					
					// 收集所有现有的text widget
					const existingTextWidgets = [];
					for (let i = node.widgets.length - 1; i >= 0; i--) {
						const widget = node.widgets[i];
						if (widget.name && widget.name.startsWith("text_")) {
							existingTextWidgets.push(node.widgets.splice(i, 1)[0]);
						}
					}
					
					// 按数字顺序添加所有需要的text widget
					for (let i = 1; i <= targetCount; i++) {
						const widgetName = `text_${i}`;
						let widget = existingTextWidgets.find(w => w.name === widgetName);
						
						if (!widget) {
							// 创建新的文本输入widget
							widget = node.addWidget("text", widgetName, "", function(v) {
								// 当widget值改变时的回调
								this.value = v;
							});
							
							// 设置widget属性
							widget.multiline = true;
							widget.inputEl && (widget.inputEl.style.height = "60px");
						} else {
							// 重新添加现有widget
							node.widgets.push(widget);
						}
					}
					
					// 将控制widget重新添加到末尾（保持原有顺序）
					controlWidgets.forEach(widget => {
						node.widgets.push(widget);
					});
				}

				nodeType.prototype.onNodeCreated = function () {
					// 重建 text_* 输入端口：仅移除多余的，添加缺失的，保留已连接的
					const rebuildTextInputs = (targetCount) => {
						// 检查节点是否已经添加到图形中
						if (!this.graph) {
							console.debug("[TextCombineMulti] Node not added to graph yet, skipping input rebuild");
							return;
						}
						
						if (!this.inputs) this.inputs = [];
						// 记录现有的 text_* 名称集合
						const existing = new Set();
						for (let i = 0; i < this.inputs.length; i++) {
							const input = this.inputs[i];
							if (input && input.name && input.name.startsWith("text_")) {
								const num = parseInt(input.name.replace("text_", ""));
								existing.add(num);
							}
						}
						// 从后往前移除多余的 text_* 输入（> targetCount）
						for (let i = this.inputs.length - 1; i >= 0; i--) {
							const input = this.inputs[i];
							if (input && input.name && input.name.startsWith("text_")) {
								const num = parseInt(input.name.replace("text_", ""));
								if (num > targetCount) {
									try {
										this.removeInput(i);
										existing.delete(num);
									} catch (e) {
										console.warn("[TextCombineMulti] Failed to remove input:", e);
									}
								}
							}
						}
						// 添加缺失的 1..targetCount
						for (let i = 1; i <= targetCount; i++) {
							if (!existing.has(i)) {
								try {
									this.addInput(`text_${i}`, "STRING");
								} catch (e) {
									console.warn("[TextCombineMulti] Failed to add input:", e);
								}
							}
						}
					};

					// 初始化时按 inputcount 重建输入
					const applyCount = (count) => {
						try { console.debug("[TextCombineMulti] applyCount", count, this.id); } catch(e) {}
						rebuildTextInputs(count);
						removeExtraTextWidgets(this, count);
						ensureTextWidgets(this, count);
						this.setSize?.(this.computeSize?.());
						this.setDirtyCanvas?.(true, true);
					};

					// 延迟初始化，等待节点添加到图形中
					const delayedInit = () => {
						const inputCountWidget = this.widgets?.find(w => w.name === "inputcount");
						if (inputCountWidget) {
							applyCount(inputCountWidget.value);
						}
					};
					
					// 如果节点已经在图形中，立即初始化
					if (this.graph) {
						delayedInit();
					} else {
						// 否则等待添加到图形后再初始化
						setTimeout(delayedInit, 0);
					}

					// Add update inputs button
					this.addWidget("button", "Update inputs", null, () => {
						const inputCountWidget = this.widgets.find(w => w.name === "inputcount");
						if (!inputCountWidget) return;
						const target_number_of_inputs = inputCountWidget.value;
						applyCount(target_number_of_inputs);
					});

					// 监听 inputcount 变化
					const inputCountWidget = this.widgets?.find(w => w.name === "inputcount");
					if (inputCountWidget) {
						const originalCallback = inputCountWidget.callback;
						inputCountWidget.callback = function(value) {
							if (originalCallback) originalCallback.call(this, value);
							applyCount(value);
						};
					}

					// 确保在添加到画布后再次应用（有些前端会重建UI）
					const onAdded = nodeType.prototype.onAdded;
					nodeType.prototype.onAdded = function() {
						if (onAdded) onAdded.apply(this, arguments);
						// 节点已添加到图形，安全地应用配置
						const w = this.widgets?.find(w => w.name === "inputcount");
						if (w) {
							console.debug("[TextCombineMulti] Node added to graph, applying count:", w.value);
							setTimeout(() => applyCount(w.value), 0);
						}
					};

					// 在反序列化或配置后也应用
					const onConfigure = nodeType.prototype.onConfigure;
					nodeType.prototype.onConfigure = function(o) {
						if (onConfigure) onConfigure.apply(this, arguments);
						
						// 恢复序列化的数量设置
						if (o && typeof o.vvl_text_count === "number") {
							setTimeout(() => applyCount(o.vvl_text_count), 0);
						} else {
							const w = this.widgets?.find(w => w.name === "inputcount");
							if (w) setTimeout(() => applyCount(w.value), 0);
						}
					};

					// 连接变化时也应用（有些前端在连接变化时会刷新端口）
					const onConnectionsChange = nodeType.prototype.onConnectionsChange;
					nodeType.prototype.onConnectionsChange = function(type, slot, isConnected, linkInfo, ioSlot) {
						if (onConnectionsChange) onConnectionsChange.apply(this, arguments);
						// 只有在节点已添加到图形时才处理
						if (this.graph) {
							const w = this.widgets?.find(w => w.name === "inputcount");
							if (w) setTimeout(() => applyCount(w.value), 0);
						}
					};

					const onConnectionsChainChange = nodeType.prototype.onConnectionsChainChange;
					nodeType.prototype.onConnectionsChainChange = function() {
						if (onConnectionsChainChange) onConnectionsChainChange.apply(this, arguments);
						// 只有在节点已添加到图形时才处理
						if (this.graph) {
							const w = this.widgets?.find(w => w.name === "inputcount");
							if (w) setTimeout(() => applyCount(w.value), 0);
						}
					};

					// 序列化/反序列化保存数量，便于恢复
					const onSerialize = nodeType.prototype.onSerialize;
					nodeType.prototype.onSerialize = function(o) {
						if (onSerialize) onSerialize.apply(this, arguments);
						const w = this.widgets?.find(w => w.name === "inputcount");
						if (w) o.vvl_text_count = w.value;
					};

					// 注意：onConfigure已经在上面重写过了，这里会导致无限递归
					// 将序列化恢复逻辑合并到现有的onConfigure中
				}
		}
	},
});