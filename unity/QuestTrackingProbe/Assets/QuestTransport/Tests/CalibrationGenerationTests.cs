using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Reflection;
using Newtonsoft.Json.Linq;
using NUnit.Framework;
using UnityEngine;

namespace QuestTeleop.Transport.Tests
{
    public class CalibrationGenerationTests
    {
        [Test]
        public void CachedAcknowledgementIsDeliveredWithoutRollingBackCurrentGeneration()
        {
            // The client is in Assembly-CSharp. Exercise the same Update queue that
            // handles real WebSocket replies without enabling the component's socket.
            var clientType = Type.GetType("QuestTeleop.Transport.QuestTransportClient, Assembly-CSharp", true);
            var gameObject = new GameObject("Transport client test");
            gameObject.SetActive(false);
            try
            {
                var client = gameObject.AddComponent(clientType);
                string session = Guid.NewGuid().ToString();
                clientType.GetProperty("SessionId").SetValue(client, session);
                var generation = clientType.GetProperty("CalibrationGeneration");
                var queue = (ConcurrentQueue<JObject>)clientType.GetField("received",
                    BindingFlags.Instance | BindingFlags.NonPublic).GetValue(client);
                var observations = new List<(long Event, long Acknowledged, long Current)>();
                clientType.GetEvent("MessageReceived").AddEventHandler(client, (Action<JObject>)(message =>
                    observations.Add(((long)message["event_id"], (long)message["calibration_generation"],
                        (long)generation.GetValue(client)))));
                JObject Ack(long eventId, long value) => new JObject {
                    ["type"] = "control_ack", ["session_id"] = session,
                    ["event_id"] = eventId, ["calibration_generation"] = value,
                };

                var original = Ack(0, 1);
                queue.Enqueue(original);
                queue.Enqueue(Ack(1, 2));
                queue.Enqueue((JObject)original.DeepClone());
                clientType.GetMethod("Update", BindingFlags.Instance | BindingFlags.NonPublic)
                    .Invoke(client, null);

                Assert.That(observations, Is.EqualTo(new[] { (0L, 1L, 1L), (1L, 2L, 2L), (0L, 1L, 2L) }));
                Assert.That(generation.GetValue(client), Is.EqualTo(2L));
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(gameObject);
            }
        }
    }
}
