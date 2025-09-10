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

				// 不需要重写序列化，让ComfyUI自己处理

				// 不需要重写configure方法，让节点自己处理

				nodeType.prototype.onNodeCreated = function () {
					
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
					const applyCount = (count, preserveConnections = false, skipWidgetRebuild = false) => {
						console.log("[TextCombineMulti] applyCount", count, this.id, "preserveConnections:", preserveConnections, "skipWidgetRebuild:", skipWidgetRebuild);
						
						// 如果不跳过widget重建，则管理widgets
						if (!skipWidgetRebuild) {
							// 管理 widgets
							removeExtraTextWidgets(this, count);
							ensureTextWidgets(this, count);
						}
						
						// 只有用户手动修改时才断开连接
						if (!preserveConnections) {
							updateInputSocketsVisibility(this, count);
						}
						
						// 让节点重新计算大小
						this.setSize?.(this.computeSize?.());
						this.setDirtyCanvas?.(true, true);
					};


					// 保存applyCount的引用到节点实例
					const nodeApplyCount = applyCount;
					
					// 重写onConfigure来处理序列化恢复
					this.onConfigure = function(o) {
						// 调用任何父类的onConfigure
						if (nodeType.prototype.onConfigure) {
							nodeType.prototype.onConfigure.call(this, o);
						}
						
						// 标记正在配置
						this._isConfiguring = true;
						
						// 延迟确保widgets已完全恢复
						setTimeout(() => {
							console.log("[TextCombineMulti] onConfigure", this.id, "widgets_values:", o?.widgets_values);
							
							// 创建一个临时存储来保存所有值
							const allValues = {};
							
							// 先从当前widgets获取值（ComfyUI可能已经部分恢复了）
							// 但不包括inputcount，因为它可能是错误的
							if (this.widgets) {
								this.widgets.forEach(w => {
									if (w.name !== 'inputcount') {
										allValues[w.name] = w.value;
									}
								});
							}
							
							// 初始化targetCount，先不要从当前widgets获取，避免干扰
							let targetCount = 2;
							
							// 如果widgets_values存在，尝试更准确地获取inputcount
							if (o && o.widgets_values && Array.isArray(o.widgets_values)) {
								console.log("[TextCombineMulti] Searching for inputcount in widgets_values...");
								
								// 在widgets_values中查找inputcount值
								// inputcount是一个数字类型，而且是整数，范围在2-20之间
								let inputcountIndex = -1;
								for (let i = 0; i < o.widgets_values.length; i++) {
									const value = o.widgets_values[i];
									console.log(`[TextCombineMulti] Index ${i}: value = ${value}, type = ${typeof value}`);
									
									// 严格检查：必须是数字类型，整数，且在有效范围内
									if (typeof value === 'number' && 
									    Number.isInteger(value) && 
									    value >= 2 && 
									    value <= 20) {
										targetCount = value;
										inputcountIndex = i;
										console.log("[TextCombineMulti] Found inputcount at index", i, "value:", value);
										// 找到第一个符合条件的就是inputcount
										break;
									}
								}
								
								// 基于找到的inputcount位置，恢复widget值
								if (inputcountIndex >= 0) {
									// inputcount前面应该有：targetCount个text widget + 1个separator
									// 所以 inputcountIndex 应该等于 targetCount + 1
									
									// 恢复text widget的值
									for (let i = 0; i < targetCount && i < o.widgets_values.length; i++) {
										allValues[`text_${i + 1}`] = o.widgets_values[i] || "";
									}
									
									// 恢复separator（在inputcount前一个位置）
									if (inputcountIndex > 0) {
										allValues.separator = o.widgets_values[inputcountIndex - 1] || "";
									}
									
									// 恢复inputcount
									allValues.inputcount = targetCount;
								}
							}
							
							console.log("[TextCombineMulti] Restoring with targetCount:", targetCount, "allValues:", allValues);
							
							// 应用正确的布局
							nodeApplyCount(targetCount, true, false);
							
							// 恢复所有widget的值
							setTimeout(() => {
								if (this.widgets) {
									this.widgets.forEach(w => {
										if (allValues[w.name] !== undefined) {
											w.value = allValues[w.name];
										}
									});
									
									// 确保inputcount是正确的数字
									const inputCountWidget = this.widgets.find(w => w.name === 'inputcount');
									if (inputCountWidget) {
										inputCountWidget.value = targetCount;
									}
								}
								this._isConfiguring = false;
							}, 10);
						}, 50);
					};
					
					// 延迟初始化，等待节点添加到图形中
					const delayedInit = () => {
						// 如果正在配置中，跳过初始化
						if (this._isConfiguring) return;
						
						const inputCountWidget = this.widgets?.find(w => w.name === "inputcount");
						if (inputCountWidget) {
							applyCount(inputCountWidget.value, true); // 初始化时保留连接
						}
					};
					
					// 如果节点已经在图形中，立即初始化
					if (this.graph) {
						// 检查是否有widgets，如果没有说明可能是新创建的
						if (!this.widgets || this.widgets.length === 0) {
							delayedInit();
						}
						// 如果已经有widgets，说明可能是从序列化恢复的，让onConfigure处理
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