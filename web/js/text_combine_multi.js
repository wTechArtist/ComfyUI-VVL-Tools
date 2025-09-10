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
					const controlWidgetNames = ["inputcount", "separator"];
					
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
					const controlWidgetNames = ["inputcount", "separator"];
					
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
					// 简单的widget管理，不触碰输入端口
					const applyCount = (count) => {
						console.log("[TextCombineMulti] applyCount", count, this.id);
						// 只管理widgets
						removeExtraTextWidgets(this, count);
						ensureTextWidgets(this, count);
						// 让节点重新计算大小
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


					// 监听 inputcount 变化
					const inputCountWidget = this.widgets?.find(w => w.name === "inputcount");
					if (inputCountWidget) {
						const originalCallback = inputCountWidget.callback;
						inputCountWidget.callback = function(value) {
							if (originalCallback) originalCallback.call(this, value);
							applyCount(value);
						};
					}

					// 节点添加到画布后应用配置
					const onAdded = nodeType.prototype.onAdded;
					nodeType.prototype.onAdded = function() {
						if (onAdded) onAdded.apply(this, arguments);
						// 节点已添加到图形，安全地应用配置
						const w = this.widgets?.find(w => w.name === "inputcount");
						if (w) {
							console.log("[TextCombineMulti] Node added to graph, applying count:", w.value);
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

					// 移除连接变化时的重建逻辑，让ComfyUI自然处理

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