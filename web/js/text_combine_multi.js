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
				// -------- Widgets 处理 --------
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

				// 重写序列化方法（在原型上）
				const originalSerialize = nodeType.prototype.serialize;
				nodeType.prototype.serialize = function() {
					const o = originalSerialize ? originalSerialize.call(this) : {};
					// 保存inputcount的值
					const inputCountWidget = this.widgets?.find(w => w.name === "inputcount");
					if (inputCountWidget) {
						o.vvl_inputcount = inputCountWidget.value;
					}
					return o;
				};

				// 重写configure方法（在原型上）
				const originalConfigure = nodeType.prototype.onConfigure;
				nodeType.prototype.onConfigure = function(o) {
					if (originalConfigure) originalConfigure.call(this, o);
					
					// 延迟处理，确保widgets已经从widgets_values恢复
					setTimeout(() => {
						const inputCountWidget = this.widgets?.find(w => w.name === "inputcount");
						if (inputCountWidget && inputCountWidget.value && this.applyCountFunction) {
							// 使用widget的实际值，它应该已经从widgets_values恢复了
							this.applyCountFunction(inputCountWidget.value, true);
						}
					}, 10); // 稍微延迟一点确保widget值已经恢复
				};

				nodeType.prototype.onNodeCreated = function () {
					const self = this;
					
					// 动态管理文本widget和输入端口的可见性
					const updateInputSocketsVisibility = (node, maxInputs) => {
						// 完全不修改输入端口的属性，只断开多余的连接
						if (!node.inputs || !Array.isArray(node.inputs)) return;
						for (let i = 0; i < node.inputs.length; i++) {
							const input = node.inputs[i];
							if (!input || typeof input.name !== "string") continue;
							if (input.name.startsWith("text_")) {
								const num = parseInt(input.name.replace("text_", ""));
								if (!Number.isNaN(num) && num > maxInputs) {
									// 只断开连接，不修改任何端口属性
									if (input.link != null && typeof node.disconnectInput === "function") {
										node.disconnectInput(i);
									}
								}
							}
						}
					};
					
					// 简单的widget管理
					const applyCount = (count, preserveConnections = false) => {
						console.log("[TextCombineMulti] applyCount", count, this.id, "preserveConnections:", preserveConnections);
						// 管理 widgets
						removeExtraTextWidgets(this, count);
						ensureTextWidgets(this, count);
						
						// 只有用户手动修改时才断开连接
						if (!preserveConnections) {
							updateInputSocketsVisibility(this, count);
						}
						
						// 让节点重新计算大小
						this.setSize?.(this.computeSize?.());
						this.setDirtyCanvas?.(true, true);
					};

					// 保存applyCount函数的引用，供onConfigure使用
					this.applyCountFunction = applyCount;

					// 延迟初始化，等待节点添加到图形中
					const delayedInit = () => {
						const inputCountWidget = this.widgets?.find(w => w.name === "inputcount");
						if (inputCountWidget) {
							applyCount(inputCountWidget.value, true); // 初始化时保留连接
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
						// 保留原始的callback引用
						const originalCallback = inputCountWidget.callback;
						inputCountWidget.callback = function(value) {
							// 先调用原始callback（这会处理输入框的关闭等逻辑）
							if (originalCallback) {
								originalCallback.apply(this, arguments);
							}
							// 应用新的count值
							applyCount(value, false); // 用户手动更改时允许断开连接
						};
					}
				}
		}
	},
});